//! Unified Scaled World Bridge (Approach 4 + Tier 3 Hybrid Soft Volumetric 3DGS):
//! - Dynamically scales to `128×128`, `256×256`, or `512×512` `A+` grids (`Vec<bool>` dirty chunks!)
//! - Builds `O(N)` Separable Top-Weighted Smoothed Terrain incrementally per dirty `16×16` chunk
//! - Maintains a live `W×H` RGBA8 Tactical Minimap texture updated incrementally per dirty chunk
//! - Emits Hard-Z GPU-instanced SDF Foliage Surfels (`PineCanopy`, `SnowPine`, `MarshBrush`, `RiverReeds`)
//! - Emits **Tier 3 Hybrid Soft Volumetric 3D Gaussian Splats (`VolSplat`)** for wildfire flames,
//!   billowing charcoal smoke, scalding steam clouds, spell trails, and bioluminescent marsh critters!
//! - Assembles 30 Hz `EntitySnapshot`s: 3 Hero Classes (`HeroKnight`, `HeroMage`, `HeroRanger`),
//!   3 Capturable `ShrineObelisk`s, Golden Relic, carved `IceRaft`s, `MarshDeer` herds, and 8 dynamic lights!

use crate::camera::CameraMatrices;
use crate::mesh_gen::{
    archetype_is_solo, MeshInstance, MeshKind, MeshVertex, CLASS_BURROWER, CLASS_EMBERSTALKER,
    CLASS_SKYWING, CLASS_STRIDER, CLASS_TIDEKIN, MESH_KIND_COUNT,
};
use crate::net::LiveEntityState;
use crate::parametric::hash2i;
use crate::surfel_gen::{SplatPlacement, SplatRef, SurfelProtoBank, SurfelProtoKind};
use crate::z16_world::Z16World;
use bytemuck::{Pod, Zeroable};
use frostmarsh_proto::{chunks_for_dim, CHUNK_SIZE, STATUS_SHIELDED};
use glam::Vec3;
use std::time::Instant;

// Species IDs matching `field-core`
pub const SP_WATER: u8 = 0;
pub const SP_ICE: u8 = 1;
pub const SP_STEAM: u8 = 2;
pub const SP_SOIL: u8 = 3;
pub const SP_ROCK: u8 = 4;
pub const SP_WOOD: u8 = 5;
pub const SP_ASH: u8 = 6;
pub const SP_SMOKE: u8 = 7;
pub const SP_OIL: u8 = 8;
pub const SP_PLANT: u8 = 9;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct VolSplat {
    pub pos_rad: [f32; 4],     // xyz = camera-relative center, w = radius_xz
    pub color_alpha: [f32; 4], // rgb = color, a = opacity
    pub params: [f32; 4],      // x = height_stretch, y = kind (0=fire,1=smoke,2=steam,3=wisp), z = phase, w = wind_strength
}

#[derive(Clone, Copy, Debug)]
pub struct DecodedAPlusCell {
    pub s1: u8,
    pub s2: u8,
    pub s2_frac: f32,
    pub flow_x: f32,
    pub flow_z: f32,
    pub temp_norm: f32,
    pub walkable: bool,
    pub wall: bool,
    pub burning: bool,
    pub freezing: bool,
    pub water_slow: bool,
    pub critter: u8,
}

#[inline]
pub fn decode_aplus_cell(raw: [u8; 4]) -> DecodedAPlusCell {
    let s1 = (raw[0] >> 4) & 0x0f;
    let s2 = raw[0] & 0x0f;
    let s2_frac = ((raw[1] >> 4) & 0x0f) as f32 / 15.0;
    let flow_4b = raw[1] & 0x0f;
    let (flow_x, flow_z) = if flow_4b == 0 {
        (0.0, 0.0)
    } else {
        let fast = flow_4b >= 9;
        let oct = if fast { (flow_4b - 9) as f32 } else { (flow_4b - 1) as f32 };
        let ang = oct * (std::f32::consts::TAU / 8.0);
        let spd = if fast { 1.8 } else { 0.85 };
        (ang.cos() * spd, ang.sin() * spd)
    };
    let temp_norm = raw[2] as f32 / 255.0;
    let hz = raw[3];
    DecodedAPlusCell {
        s1,
        s2,
        s2_frac,
        flow_x,
        flow_z,
        temp_norm,
        walkable: (hz & (1 << 0)) != 0,
        wall: (hz & (1 << 1)) != 0,
        burning: (hz & (1 << 2)) != 0,
        freezing: (hz & (1 << 3)) != 0,
        water_slow: (hz & (1 << 4)) != 0,
        critter: (hz >> 5) & 0x07,
    }
}

fn species_albedo(sp: u8) -> [f32; 3] {
    match sp {
        SP_WATER => [0.16, 0.44, 0.68],
        SP_ICE => [0.72, 0.90, 0.98],
        SP_STEAM => [0.84, 0.90, 0.95],
        SP_SOIL => [0.33, 0.45, 0.28],
        SP_ROCK => [0.39, 0.45, 0.57],
        SP_WOOD => [0.30, 0.43, 0.26],
        SP_ASH => [0.22, 0.20, 0.24],
        SP_SMOKE => [0.30, 0.29, 0.34],
        SP_OIL => [0.18, 0.12, 0.26],
        SP_PLANT => [0.26, 0.58, 0.31],
        _ => [0.24, 0.30, 0.28],
    }
}

fn cell_height_and_color(
    c: &DecodedAPlusCell,
    gx: usize,
    gy: usize,
    grid_w: usize,
    grid_h: usize,
) -> (f32, [f32; 3], bool, f32) {
    let is_ice = c.s1 == SP_ICE || (c.s2 == SP_ICE && c.s2_frac > 0.30);
    let is_bridge = c.s1 == SP_WOOD && !c.wall && c.walkable;
    let is_water = !is_ice
        && !is_bridge
        && ((c.s1 == SP_WATER && c.s2 != SP_SOIL)
            || (c.water_slow && !c.walkable && c.s1 != SP_ROCK));

    let nx = gx as f32 / grid_w.max(1) as f32;
    let ny = gy as f32 / grid_h.max(1) as f32;
    let gentle_hill = 0.58 * (nx * 11.0).sin() * (ny * 9.0).cos()
        + 0.28 * (nx * 23.0 + ny * 17.0).sin()
        + (1.0 - ny) * 0.95;

    // Keep terrain height smooth & continuous: never extrude sharp 1-cell pyramid spikes on `c.wall`!
    // Discrete boulders (`RockOutcrop`) handle sharp rock silhouette geometry cleanly above the smooth hills.
    let h = if c.s1 == SP_ROCK {
        if gx < 4 || gx + 4 >= grid_w || gy < 4 || gy + 4 >= grid_h {
            4.4 + gentle_hill * 0.55
        } else {
            2.15 + gentle_hill * 0.50
        }
    } else if is_bridge {
        0.72
    } else if is_ice {
        0.34
    } else if is_water {
        -0.65
    } else if c.s1 == SP_OIL && c.s2 == SP_WATER {
        -0.48
    } else if c.s1 == SP_OIL || (c.s2 == SP_OIL && c.s2_frac > 0.25) {
        0.72 + gentle_hill * 0.28
    } else if c.s1 == SP_ASH {
        0.92 + gentle_hill * 0.55
    } else {
        1.18 + gentle_hill
    };

    let c1 = species_albedo(c.s1);
    let c2 = if c.s2 < 10 { species_albedo(c.s2) } else { c1 };
    let w2 = if c.s2 < 10 { c.s2_frac * 0.65 } else { 0.0 };
    let mut rgb = [
        c1[0] * (1.0 - w2) + c2[0] * w2,
        c1[1] * (1.0 - w2) + c2[1] * w2,
        c1[2] * (1.0 - w2) + c2[2] * w2,
    ];

    if is_bridge {
        // Warm weathered timber planks spanning the fjord
        let plank = if (gx + gy) % 2 == 0 { 1.0 } else { 0.88 };
        rgb = [0.56 * plank, 0.37 * plank, 0.22 * plank];
    } else if c.s1 == SP_OIL || (c.s2 == SP_OIL && c.s2_frac > 0.22) {
        let swirl = hash2i(gx as i32 * 13 + 5, gy as i32 * 29 + 7) * 0.45
            + 0.55 * (0.5 + 0.5 * ((gx as f32 * 0.14).sin() * (gy as f32 * 0.17).cos()));
        rgb = [
            rgb[0] * 0.78 + 0.12 * swirl,
            rgb[1] * 0.78 + 0.06 * (1.0 - swirl),
            rgb[2] * 0.78 + 0.22 * swirl,
        ];
    } else if (c.s1 == SP_SOIL && c.s2 == SP_WATER) || (c.s1 == SP_WATER && c.s2 == SP_SOIL) {
        rgb = [0.30, 0.24, 0.20];
    }

    let burn_t = if c.burning {
        0.92
    } else if c.temp_norm > 0.26 {
        ((c.temp_norm - 0.26) / 0.36).clamp(0.0, 0.85)
    } else {
        0.0
    };

    (h, rgb, is_water, burn_t)
}

#[derive(Clone, Debug)]
pub struct ChunkSurfelItem {
    pub kind: SurfelProtoKind,
    pub world_pos: Vec3,
    pub yaw: f32,
    pub scale: f32,
    pub tint: [f32; 3],
    pub wind_phase: f32,
    pub glow_mul: f32,
}

#[derive(Clone, Debug)]
pub struct ChunkVolItem {
    pub world_pos: Vec3,
    pub radius: f32,
    pub height_stretch: f32,
    pub color: [f32; 3],
    pub alpha: f32,
    pub kind: f32, // 0=fire, 1=smoke, 2=steam, 3=wisp
    pub phase: f32,
}

#[derive(Clone, Debug)]
pub struct ChunkPropItem {
    pub kind: MeshKind,
    pub world_pos: Vec3,
    pub yaw: f32,
    pub scale: Vec3,
    pub tint: [f32; 3],
    pub glow: f32,
}

#[derive(Default)]
pub struct ChunkCache {
    pub center_xz: (f32, f32),
    pub terrain_verts: Vec<MeshVertex>,
    pub terrain_idxs: Vec<u32>,
    pub props: Vec<ChunkPropItem>,
    pub surfels: Vec<ChunkSurfelItem>,
    pub volumetrics: Vec<ChunkVolItem>,
    pub fire_hotspots: Vec<(Vec3, f32)>,
    pub waterfall_points: Vec<Vec3>,
}

#[derive(Clone, Default, Debug)]
pub struct WorldSummaryStats {
    pub wood_cells: usize,
    pub plant_cells: usize,
    pub water_cells: usize,
    pub ice_cells: usize,
    pub steam_cells: usize,
    pub oil_cells: usize,
    pub rock_cells: usize,
    pub ash_cells: usize,
    pub burning_cells: usize,
    pub freezing_cells: usize,
    pub critter_cells: usize,
    pub total_pine_trees: usize,
    pub total_brush_clusters: usize,
    pub total_volumetric_splats: usize,
    pub total_wildlife_deer: usize,
}

pub struct LiveApproach4World {
    pub grid_w: usize,
    pub grid_h: usize,
    pub grid_d: usize,
    pub chunks_x: usize,
    pub chunks_y: usize,
    pub decoded: Vec<DecodedAPlusCell>,
    pub decoded_l2: Vec<DecodedAPlusCell>,
    pub decoded_l3: Vec<DecodedAPlusCell>,
    pub height_grid: Vec<f32>,
    pub smooth_normals: Vec<[f32; 3]>,
    pub minimap_rgba: Vec<u8>,
    pub minimap_dirty: bool,
    pub chunks: Vec<ChunkCache>,
    pub chunks_l2: Vec<ChunkCache>,
    pub chunks_l3: Vec<ChunkCache>,
    pub summary: WorldSummaryStats,
}

impl LiveApproach4World {
    pub fn new(grid_w: usize, grid_h: usize, grid_d: usize) -> Self {
        let chunks_x = chunks_for_dim(grid_w);
        let chunks_y = chunks_for_dim(grid_h);
        let n = grid_w * grid_h;
        let air = decode_aplus_cell([0xff, 0, 26, 1]);
        Self {
            grid_w,
            grid_h,
            grid_d: grid_d.max(1),
            chunks_x,
            chunks_y,
            decoded: vec![air; n],
            decoded_l2: vec![air; n],
            decoded_l3: vec![air; n],
            height_grid: vec![0.9; n],
            smooth_normals: vec![[0.0, 1.0, 0.0]; n],
            minimap_rgba: vec![32u8; n * 4],
            minimap_dirty: true,
            chunks: (0..(chunks_x * chunks_y)).map(|_| ChunkCache::default()).collect(),
            chunks_l2: (0..(chunks_x * chunks_y)).map(|_| ChunkCache::default()).collect(),
            chunks_l3: (0..(chunks_x * chunks_y)).map(|_| ChunkCache::default()).collect(),
            summary: WorldSummaryStats::default(),
        }
    }

    pub fn surf_layer(&self) -> usize {
        if self.grid_d >= 2 { 1 } else { 0 }
    }

    pub fn resize_if_needed(&mut self, w: usize, h: usize, d: usize) {
        let d = d.max(1);
        if w > 0 && h > 0 && (self.grid_w != w || self.grid_h != h || self.grid_d != d) {
            *self = Self::new(w, h, d);
        }
    }

    #[inline]
    pub fn grid_to_world_xz(&self, gx: f32, gy: f32) -> (f32, f32) {
        (gx - (self.grid_w as f32) * 0.5, gy - (self.grid_h as f32) * 0.5)
    }

    #[inline]
    pub fn world_xz_to_grid(&self, wx: f32, wz: f32) -> (f32, f32) {
        (wx + (self.grid_w as f32) * 0.5, wz + (self.grid_h as f32) * 0.5)
    }

    /// Incrementally update only the dirty `(layer, chunk_idx)` 3D volumetric chunks across all 4 strata!
    pub fn update_dirty_chunks(
        &mut self,
        aplus_layers: &[Vec<[u8; 4]>],
        dirty_items: &[(usize, usize)],
    ) -> (usize, f64) {
        if dirty_items.is_empty() {
            return (0, 0.0);
        }
        let t0 = Instant::now();
        let gw = self.grid_w;
        let gh = self.grid_h;
        let cx_dim = self.chunks_x;
        let cy_dim = self.chunks_y;
        let surf_l = self.surf_layer();
        let mut rebuilt = 0usize;

        let mut surf_dirty: Vec<usize> = Vec::new();
        let mut l2_dirty: Vec<usize> = Vec::new();
        let mut l3_dirty: Vec<usize> = Vec::new();

        for &(layer, ci) in dirty_items {
            if ci >= cx_dim * cy_dim {
                continue;
            }
            if layer == surf_l {
                surf_dirty.push(ci);
            } else if layer == 2 && self.grid_d >= 3 {
                l2_dirty.push(ci);
            } else if layer == 3 && self.grid_d >= 4 {
                l3_dirty.push(ci);
            }
        }

        // --- Surface layer (L1) ---
        if !surf_dirty.is_empty() && surf_l < aplus_layers.len() {
            let aplus_raw = &aplus_layers[surf_l];
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (gw, gh, 0usize, 0usize);
            for &ci in &surf_dirty {
                let cx = ci % cx_dim;
                let cy = ci / cx_dim;
                let x0 = cx * CHUNK_SIZE;
                let y0 = cy * CHUNK_SIZE;
                min_x = min_x.min(x0);
                min_y = min_y.min(y0);
                max_x = max_x.max((x0 + CHUNK_SIZE).min(gw));
                max_y = max_y.max((y0 + CHUNK_SIZE).min(gh));

                for dy in 0..CHUNK_SIZE {
                    let gy = y0 + dy;
                    if gy >= gh { continue; }
                    for dx in 0..CHUNK_SIZE {
                        let gx = x0 + dx;
                        if gx >= gw { continue; }
                        let idx = gy * gw + gx;
                        let c = decode_aplus_cell(aplus_raw[idx]);
                        let (h, rgb, _, burn_t) = cell_height_and_color(&c, gx, gy, gw, gh);
                        self.decoded[idx] = c;
                        self.height_grid[idx] = h;

                        let (mut mr, mut mg, mut mb) = (rgb[0], rgb[1], rgb[2]);
                        if c.burning || burn_t > 0.55 {
                            mr = 1.0; mg = 0.54; mb = 0.14;
                        } else if c.s1 == SP_WOOD {
                            mr = 0.14; mg = 0.36; mb = 0.22;
                        } else if c.s1 == SP_STEAM {
                            mr = 0.85; mg = 0.92; mb = 0.98;
                        } else if c.critter >= 3 {
                            mr = 0.82; mg = 0.96; mb = 0.42;
                        }
                        let off = idx * 4;
                        self.minimap_rgba[off] = (mr.clamp(0.0, 1.0) * 255.0) as u8;
                        self.minimap_rgba[off + 1] = (mg.clamp(0.0, 1.0) * 255.0) as u8;
                        self.minimap_rgba[off + 2] = (mb.clamp(0.0, 1.0) * 255.0) as u8;
                        self.minimap_rgba[off + 3] = 255;
                    }
                }
                rebuilt += 1;
            }
            self.minimap_dirty = true;

            let (rx0, ry0, rx1, ry1) = if surf_dirty.len() >= (cx_dim * cy_dim) / 2 {
                (0usize, 0usize, gw, gh)
            } else {
                (min_x.saturating_sub(3), min_y.saturating_sub(3), (max_x + 3).min(gw), (max_y + 3).min(gh))
            };
            self.smooth_height_grid_region(rx0, ry0, rx1, ry1);
            self.compute_separable_top_weighted_normals(rx0, ry0, rx1, ry1);

            for &ci in &surf_dirty {
                self.rebuild_chunk(ci % cx_dim, ci / cx_dim);
            }
        }

        // --- Layer 2 (Highland Basalt Plateaus, Overhanging Skybridges, Alpine Lakes & 3D Waterfalls!) ---
        if !l2_dirty.is_empty() && 2 < aplus_layers.len() {
            let raw2 = &aplus_layers[2];
            // Pass 1: decode ALL dirty L2 chunks BEFORE meshing so neighbor checks across 16×16 chunk seams
            // never see stale air and never emit phantom internal cliff/waterfall ribs!
            for &ci in &l2_dirty {
                let cx = ci % cx_dim;
                let cy = ci / cx_dim;
                let x0 = cx * CHUNK_SIZE;
                let y0 = cy * CHUNK_SIZE;
                for dy in 0..CHUNK_SIZE {
                    let gy = y0 + dy;
                    if gy >= gh { continue; }
                    for dx in 0..CHUNK_SIZE {
                        let gx = x0 + dx;
                        if gx >= gw { continue; }
                        let idx = gy * gw + gx;
                        let c = decode_aplus_cell(raw2[idx]);
                        self.decoded_l2[idx] = c;
                        // Tint minimap subtly where Layer 2 Highland/Skybridge exists!
                        if c.s1 < 10 {
                            let off = idx * 4;
                            if c.s1 == SP_WATER {
                                self.minimap_rgba[off] = 95;
                                self.minimap_rgba[off + 1] = 195;
                                self.minimap_rgba[off + 2] = 248;
                            } else if c.s1 == SP_WOOD && !c.wall {
                                self.minimap_rgba[off] = 178;
                                self.minimap_rgba[off + 1] = 128;
                                self.minimap_rgba[off + 2] = 74;
                            } else if c.s1 == SP_SOIL {
                                self.minimap_rgba[off] = 138;
                                self.minimap_rgba[off + 1] = 148;
                                self.minimap_rgba[off + 2] = 166;
                            }
                        }
                    }
                }
            }
            // Pass 2: rebuild L2 chunk meshes (plus immediate 4-neighbors on incremental updates to heal seams)
            let mut to_rebuild = l2_dirty.clone();
            if l2_dirty.len() < (cx_dim * cy_dim) / 2 {
                for &ci in &l2_dirty {
                    let cx = ci % cx_dim;
                    let cy = ci / cx_dim;
                    if cx > 0 { to_rebuild.push(cy * cx_dim + (cx - 1)); }
                    if cx + 1 < cx_dim { to_rebuild.push(cy * cx_dim + (cx + 1)); }
                    if cy > 0 { to_rebuild.push((cy - 1) * cx_dim + cx); }
                    if cy + 1 < cy_dim { to_rebuild.push((cy + 1) * cx_dim + cx); }
                }
                to_rebuild.sort_unstable();
                to_rebuild.dedup();
            }
            for &ci in &to_rebuild {
                self.rebuild_layer2_chunk(ci % cx_dim, ci / cx_dim);
                rebuilt += 1;
            }
            self.minimap_dirty = true;
        }

        // --- Layer 3 (Upper Atmosphere: Rising 3D Volumetric Steam, Smoke & Ember Plumes!) ---
        if !l3_dirty.is_empty() && 3 < aplus_layers.len() {
            let raw3 = &aplus_layers[3];
            for &ci in &l3_dirty {
                let cx = ci % cx_dim;
                let cy = ci / cx_dim;
                let x0 = cx * CHUNK_SIZE;
                let y0 = cy * CHUNK_SIZE;
                for dy in 0..CHUNK_SIZE {
                    let gy = y0 + dy;
                    if gy >= gh { continue; }
                    for dx in 0..CHUNK_SIZE {
                        let gx = x0 + dx;
                        if gx >= gw { continue; }
                        let idx = gy * gw + gx;
                        self.decoded_l3[idx] = decode_aplus_cell(raw3[idx]);
                    }
                }
                self.rebuild_layer3_chunk(cx, cy);
                rebuilt += 1;
            }
        }

        self.recompute_summary();
        (rebuilt, t0.elapsed().as_secs_f64() * 1000.0)
    }

    fn push_box_quad(
        verts: &mut Vec<MeshVertex>,
        idxs: &mut Vec<u32>,
        p0: [f32; 3],
        p1: [f32; 3],
        p2: [f32; 3],
        p3: [f32; 3],
        n: [f32; 3],
        cm: [f32; 4],
        extra: [f32; 4],
    ) {
        let base = verts.len() as u32;
        for p in [p0, p1, p2, p3] {
            verts.push(MeshVertex { pos: p, normal: n, color_mat: cm, extra });
        }
        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Build true 3D Layer 2 geometry: elevated basalt plateaus, overhanging Stone Sky-Arch &
    /// Timber Skybridge decks (with open fjord flowing underneath!), Highland Alpine Lakes,
    /// animated plunging 3D Waterfall Curtains, and Highland Alpine Pine Surfels!
    fn rebuild_layer2_chunk(&mut self, cx: usize, cy: usize) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        let ci = cy * self.chunks_x + cx;
        let x0 = cx * CHUNK_SIZE;
        let y0 = cy * CHUNK_SIZE;
        let center_xz = self.grid_to_world_xz((x0 + 8) as f32, (y0 + 8) as f32);

        let mut cache = ChunkCache {
            center_xz,
            terrain_verts: Vec::with_capacity(512),
            terrain_idxs: Vec::with_capacity(1024),
            props: Vec::with_capacity(32),
            surfels: Vec::with_capacity(64),
            volumetrics: Vec::with_capacity(64),
            fire_hotspots: Vec::with_capacity(2),
            waterfall_points: Vec::with_capacity(4),
        };

        let mut wf_sum = Vec3::ZERO;
        let mut wf_cnt = 0.0f32;
        let mut l2_fire_sum = Vec3::ZERO;
        let mut l2_fire_cnt = 0.0f32;

        let is_solid_l2 = |gx: isize, gy: isize| -> bool {
            if gx < 0 || gy < 0 || gx >= gw as isize || gy >= gh as isize {
                return false;
            }
            let c = &self.decoded_l2[gy as usize * gw + gx as usize];
            c.s1 < 10 && c.s1 != SP_SMOKE && c.s1 != SP_STEAM
        };

        for dy in 0..CHUNK_SIZE {
            let gy = y0 + dy;
            if gy >= gh { continue; }
            for dx in 0..CHUNK_SIZE {
                let gx = x0 + dx;
                if gx >= gw { continue; }
                let idx = gy * gw + gx;
                let c = &self.decoded_l2[idx];
                if c.s1 >= 10 {
                    continue; // open air on Layer 2
                }
                let (wx0, wz0) = self.grid_to_world_xz(gx as f32, gy as f32);
                let (wx, wz) = (wx0 + 0.5, wz0 + 0.5);
                let j0 = hash2i(gx as i32 * 23 + 7, gy as i32 * 37 + 13);
                let j1 = hash2i(gx as i32 * 41 - 3, gy as i32 * 17 + 5);

                // Mid-air rising Steam / Smoke on Layer 2 -> Delicate Translucent Volumetric 3DGS!
                if c.s1 == SP_STEAM || c.s1 == SP_SMOKE {
                    if j0 < 0.14 {
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(wx, 8.8 + j1 * 1.6, wz),
                            radius: 1.05 + j1 * 0.50,
                            height_stretch: 1.12,
                            color: if c.s1 == SP_STEAM { [0.80, 0.90, 0.98] } else { [0.22, 0.20, 0.28] },
                            alpha: if c.s1 == SP_STEAM { 0.22 } else { 0.24 },
                            kind: if c.s1 == SP_STEAM { 2.0 } else { 1.0 },
                            phase: j0 * 14.0,
                        });
                    }
                    continue;
                }

                // Determine Layer 2 vertical slab height & overhang thickness!
                let is_skybridge_wood = c.s1 == SP_WOOD && !c.wall && c.walkable;
                let is_water_l2 = c.s1 == SP_WATER;
                let surf_sp = self.decoded[idx].s1;
                let is_overhang_arch = !is_water_l2
                    && (surf_sp == SP_WATER
                        || surf_sp == SP_ICE
                        || surf_sp == SP_OIL
                        || surf_sp == SP_STEAM
                        || self.height_grid[idx] < 0.45);

                // Count adjacent L2 water neighbors so only contiguous highland lakes/rivers form waterfalls
                let mut water_nbs = 0u8;
                if is_water_l2 {
                    for (ox, oy) in [(1isize, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nx = gx as isize + ox;
                        let ny = gy as isize + oy;
                        if nx >= 0 && ny >= 0 && nx < gw as isize && ny < gh as isize {
                            if self.decoded_l2[ny as usize * gw + nx as usize].s1 == SP_WATER {
                                water_nbs += 1;
                            }
                        }
                    }
                    if water_nbs == 0 {
                        // Skip isolated 1-voxel L2 rain droplets so they never spawn floating water blocks!
                        continue;
                    }
                }

                // Towering monumental Layer 2 elevation (~2.5x hero height) with ~6.0u clearance under arches!
                let top_y = if is_water_l2 { 7.85 } else if is_skybridge_wood { 8.15 } else { 8.35 };
                // Overhanging skybridges & stone arches are 1.35u thick so heroes walk freely underneath!
                // Solid basalt plateaus extend all the way down into the ground (`bot_y = 0.4`)!
                let bot_y = if is_overhang_arch || is_skybridge_wood { 6.85 } else { 0.40 };

                let (top_col, side_col, mat_id) = if is_water_l2 {
                    ([0.24, 0.64, 0.92], [0.46, 0.80, 0.98], 3.0)
                } else if is_skybridge_wood {
                    let p = if (gx + gy) % 2 == 0 { 1.0 } else { 0.86 };
                    ([0.62 * p, 0.41 * p, 0.24 * p], [0.38, 0.25, 0.16], 4.0)
                } else if c.s1 == SP_OIL {
                    ([0.24, 0.14, 0.34], [0.34, 0.36, 0.46], 9.0)
                } else if c.s1 == SP_ICE {
                    ([0.76, 0.93, 1.0], [0.58, 0.82, 0.96], 6.0)
                } else if is_overhang_arch {
                    ([0.56, 0.60, 0.70], [0.38, 0.42, 0.54], 1.0)
                } else {
                    // Highland basalt plateau top (frost-dusted alpine turf/stone)
                    let snowy_t = (1.0 - (gy as f32 / gh as f32) * 2.2).clamp(0.0, 1.0);
                    let r = 0.36 * (1.0 - snowy_t) + 0.74 * snowy_t;
                    let g = 0.46 * (1.0 - snowy_t) + 0.84 * snowy_t;
                    let b = 0.42 * (1.0 - snowy_t) + 0.92 * snowy_t;
                    ([r, g, b], [0.34, 0.38, 0.50], 9.0)
                };

                let extra = [c.temp_norm, if c.burning { 0.75 } else { 0.0 }, c.flow_x, c.flow_z];
                // Top face (+Y)
                Self::push_box_quad(
                    &mut cache.terrain_verts,
                    &mut cache.terrain_idxs,
                    [wx0, top_y, wz0],
                    [wx0, top_y, wz0 + 1.0],
                    [wx0 + 1.0, top_y, wz0 + 1.0],
                    [wx0 + 1.0, top_y, wz0],
                    [0.0, 1.0, 0.0],
                    [top_col[0], top_col[1], top_col[2], mat_id],
                    extra,
                );

                // Emit cliff wall faces OR animated 3D Waterfall Curtains on exposed rims!
                let nbs = [
                    (gx as isize + 1, gy as isize, [1.0, 0.0, 0.0], [wx0 + 1.0, bot_y, wz0 + 1.0], [wx0 + 1.0, top_y, wz0 + 1.0], [wx0 + 1.0, top_y, wz0], [wx0 + 1.0, bot_y, wz0]),
                    (gx as isize - 1, gy as isize, [-1.0, 0.0, 0.0], [wx0, bot_y, wz0], [wx0, top_y, wz0], [wx0, top_y, wz0 + 1.0], [wx0, bot_y, wz0 + 1.0]),
                    (gx as isize, gy as isize + 1, [0.0, 0.0, 1.0], [wx0, bot_y, wz0 + 1.0], [wx0, top_y, wz0 + 1.0], [wx0 + 1.0, top_y, wz0 + 1.0], [wx0 + 1.0, bot_y, wz0 + 1.0]),
                    (gx as isize, gy as isize - 1, [0.0, 0.0, -1.0], [wx0 + 1.0, bot_y, wz0], [wx0 + 1.0, top_y, wz0], [wx0, top_y, wz0], [wx0, bot_y, wz0]),
                ];
                let mut waterfall_edge = false;
                for (nx, ny, norm, p0, p1, p2, p3) in nbs {
                    if !is_solid_l2(nx, ny) {
                        if is_water_l2 {
                            // Only emit a plunging 3D Waterfall Curtain when pouring off a true rim
                            // (contiguous water body pouring over open fjord/ice/cliff drop, not shallow puddles!)
                            let nb_in_bounds = nx >= 0 && ny >= 0 && nx < gw as isize && ny < gh as isize;
                            let nb_surf_low = nb_in_bounds && {
                                let nidx = ny as usize * gw + nx as usize;
                                let ns = self.decoded[nidx].s1;
                                ns == SP_WATER || ns == SP_ICE || self.height_grid[nidx] < 0.75
                            };
                            if water_nbs >= 2 && nb_surf_low {
                                let mut wp0 = p0; let mut wp3 = p3;
                                wp0[1] = 0.05; wp3[1] = 0.05;
                                Self::push_box_quad(
                                    &mut cache.terrain_verts,
                                    &mut cache.terrain_idxs,
                                    wp0, p1, p2, wp3,
                                    norm,
                                    [0.44, 0.82, 0.98, 3.0],
                                    [c.temp_norm, 0.0, 0.0, 2.2],
                                );
                                waterfall_edge = true;
                            }
                        } else {
                            Self::push_box_quad(
                                &mut cache.terrain_verts,
                                &mut cache.terrain_idxs,
                                p0, p1, p2, p3,
                                norm,
                                [side_col[0], side_col[1], side_col[2], 1.0],
                                extra,
                            );
                        }
                    }
                }

                // Delicate 3D waterfall splash mist at the base of the cascade (keeps stone arch crisp & visible!)
                if waterfall_edge {
                    wf_sum += Vec3::new(wx, 2.2, wz);
                    wf_cnt += 1.0;
                    if j0 < 0.22 {
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(wx, 0.85 + j1 * 0.65, wz),
                            radius: 0.92 + j1 * 0.40,
                            height_stretch: 1.10,
                            color: [0.76, 0.90, 0.99],
                            alpha: 0.24,
                            kind: 2.0,
                            phase: j0 * 16.0,
                        });
                    }
                }

                // Burning Layer 2 Timber Skybridge / Oil Cascade -> Soft Volumetric Fire!
                if c.burning {
                    l2_fire_sum += Vec3::new(wx, top_y + 0.85, wz);
                    l2_fire_cnt += 1.0;
                    if j0 < 0.55 {
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(wx, top_y + 0.85, wz),
                            radius: 0.85 + j1 * 0.45,
                            height_stretch: 1.85,
                            color: [1.35, 0.50, 0.10],
                            alpha: 0.76,
                            kind: 0.0,
                            phase: j0 * 15.0,
                        });
                    }
                }

                // Majestic spaced Highland Snow-Pine Surfels perched up on Layer 2 plateaus!
                if !is_skybridge_wood && c.s1 == SP_WOOD && c.wall && (gx % 3 == 0 && gy % 3 == 0) {
                    let sc = 1.75 + j0 * 0.85;
                    let jx = (j0 - 0.5) * 1.4;
                    let jz = (j1 - 0.5) * 1.4;
                    cache.props.push(ChunkPropItem {
                        kind: MeshKind::TreeTrunk,
                        world_pos: Vec3::new(wx + jx, top_y - 0.12, wz + jz),
                        yaw: j0 * 6.28,
                        scale: Vec3::new(sc * 0.92, sc * 1.05, sc * 0.92),
                        tint: [1.0, 1.0, 1.0],
                        glow: 0.0,
                    });
                    cache.surfels.push(ChunkSurfelItem {
                        kind: SurfelProtoKind::SnowPine,
                        world_pos: Vec3::new(wx + jx, top_y, wz + jz),
                        yaw: j1 * 6.28,
                        scale: sc,
                        tint: [0.96, 0.99, 1.04],
                        wind_phase: gx as f32 * 0.4 + gy as f32 * 0.5,
                        glow_mul: 0.0,
                    });
                }
            }
        }
        if wf_cnt > 0.0 {
            cache.waterfall_points.push(wf_sum / wf_cnt);
        }
        if l2_fire_cnt > 0.0 {
            cache.fire_hotspots.push((l2_fire_sum / l2_fire_cnt, l2_fire_cnt));
        }
        self.chunks_l2[ci] = cache;
    }

    /// Build Layer 3 Upper-Atmosphere Soft Volumetric 3D Gaussian Plume Clouds (`y = +7.0..+10.5`)!
    fn rebuild_layer3_chunk(&mut self, cx: usize, cy: usize) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        let ci = cy * self.chunks_x + cx;
        let x0 = cx * CHUNK_SIZE;
        let y0 = cy * CHUNK_SIZE;
        let center_xz = self.grid_to_world_xz((x0 + 8) as f32, (y0 + 8) as f32);

        let mut cache = ChunkCache {
            center_xz,
            ..Default::default()
        };

        for dy in 0..CHUNK_SIZE {
            let gy = y0 + dy;
            if gy >= gh { continue; }
            for dx in 0..CHUNK_SIZE {
                let gx = x0 + dx;
                if gx >= gw { continue; }
                let idx = gy * gw + gx;
                let c = &self.decoded_l3[idx];
                if c.s1 >= 10 {
                    continue;
                }
                let j0 = hash2i(gx as i32 * 29 + 11, gy as i32 * 13 + 7);
                let j1 = hash2i(gx as i32 * 53 - 9, gy as i32 * 31 + 3);
                if j0 > 0.14 {
                    continue;
                }
                let (wx, wz) = self.grid_to_world_xz(gx as f32 + 0.5, gy as f32 + 0.5);
                if c.s1 == SP_STEAM || c.s2 == SP_STEAM {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, 14.2 + j1 * 3.2, wz),
                        radius: 1.75 + j1 * 0.85,
                        height_stretch: 0.82,
                        color: [0.80, 0.90, 0.98],
                        alpha: 0.17,
                        kind: 2.0,
                        phase: j0 * 20.0,
                    });
                } else if c.s1 == SP_SMOKE || c.s2 == SP_SMOKE {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, 15.0 + j1 * 3.5, wz),
                        radius: 1.95 + j0 * 1.05,
                        height_stretch: 0.88,
                        color: [0.20, 0.18, 0.28],
                        alpha: 0.22,
                        kind: 1.0,
                        phase: j1 * 18.0,
                    });
                }
            }
        }
        self.chunks_l3[ci] = cache;
    }

    /// Fast separable 3×3 height smoothing pass over dirty terrain regions so isolated
    /// rock/border/species transitions form organic rolling hills and NEVER 1-cell spikes!
    fn smooth_height_grid_region(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        let gw = self.grid_w;
        let rw = x1.saturating_sub(x0);
        let rh = y1.saturating_sub(y0);
        if rw == 0 || rh == 0 {
            return;
        }
        let mut tmp = vec![0.0f32; rw * rh];
        for ly in 0..rh {
            let gy = y0 + ly;
            let row_g = gy * gw;
            for lx in 0..rw {
                let gx = x0 + lx;
                let xm = gx.saturating_sub(1);
                let xp = (gx + 1).min(gw - 1);
                tmp[ly * rw + lx] = 0.25 * self.height_grid[row_g + xm]
                    + 0.50 * self.height_grid[row_g + gx]
                    + 0.25 * self.height_grid[row_g + xp];
            }
        }
        for ly in 0..rh {
            let gy = y0 + ly;
            let ym = ly.saturating_sub(1);
            let yp = (ly + 1).min(rh - 1);
            for lx in 0..rw {
                let gx = x0 + lx;
                let idx = gy * gw + gx;
                // Keep bridges flat and water basins depressed cleanly while smoothing land/banks
                let c = &self.decoded[idx];
                let is_bridge = c.s1 == SP_WOOD && !c.wall && c.walkable;
                if is_bridge {
                    continue;
                }
                self.height_grid[idx] = 0.25 * tmp[ym * rw + lx]
                    + 0.50 * tmp[ly * rw + lx]
                    + 0.25 * tmp[yp * rw + lx];
            }
        }
    }

    fn compute_separable_top_weighted_normals(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        let rw = x1.saturating_sub(x0);
        let rh = y1.saturating_sub(y0);
        if rw == 0 || rh == 0 {
            return;
        }
        let mut raw_n = vec![[0.0f32; 3]; rw * rh];
        for ly in 0..rh {
            let gy = y0 + ly;
            let ym = gy.saturating_sub(1);
            let yp = (gy + 1).min(gh - 1);
            for lx in 0..rw {
                let gx = x0 + lx;
                let xm = gx.saturating_sub(1);
                let xp = (gx + 1).min(gw - 1);
                let hx = self.height_grid[gy * gw + xp] - self.height_grid[gy * gw + xm];
                let hz = self.height_grid[yp * gw + gx] - self.height_grid[ym * gw + gx];
                raw_n[ly * rw + lx] = [-hx, 2.0 * 1.45, -hz];
            }
        }

        let radius = 1usize;
        let mut tmp = vec![[0.0f32; 3]; rw * rh];

        for ly in 0..rh {
            let row = ly * rw;
            let (mut sx, mut sy, mut sz) = (0.0f32, 0.0f32, 0.0f32);
            for q in 0..=radius.min(rw - 1) {
                let v = raw_n[row + q];
                sx += v[0];
                sy += v[1];
                sz += v[2];
            }
            for lx in 0..rw {
                tmp[row + lx] = [sx, sy, sz];
                let add = lx + radius + 1;
                if add < rw {
                    let v = raw_n[row + add];
                    sx += v[0];
                    sy += v[1];
                    sz += v[2];
                }
                if lx >= radius {
                    let v = raw_n[row + lx - radius];
                    sx -= v[0];
                    sy -= v[1];
                    sz -= v[2];
                }
            }
        }

        for lx in 0..rw {
            let (mut sx, mut sy, mut sz) = (0.0f32, 0.0f32, 0.0f32);
            for q in 0..=radius.min(rh - 1) {
                let v = tmp[q * rw + lx];
                sx += v[0];
                sy += v[1];
                sz += v[2];
            }
            for ly in 0..rh {
                let len = (sx * sx + sy * sy + sz * sz).sqrt();
                self.smooth_normals[(y0 + ly) * gw + (x0 + lx)] = if len < 1e-4 {
                    [0.0, 1.0, 0.0]
                } else {
                    [sx / len, sy / len, sz / len]
                };
                let add = ly + radius + 1;
                if add < rh {
                    let v = tmp[add * rw + lx];
                    sx += v[0];
                    sy += v[1];
                    sz += v[2];
                }
                if ly >= radius {
                    let v = tmp[(ly - radius) * rw + lx];
                    sx -= v[0];
                    sy -= v[1];
                    sz -= v[2];
                }
            }
        }
    }

    fn rebuild_chunk(&mut self, cx: usize, cy: usize) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        let ci = cy * self.chunks_x + cx;
        let x0 = cx * CHUNK_SIZE;
        let y0 = cy * CHUNK_SIZE;
        let x1 = (x0 + CHUNK_SIZE).min(gw - 1);
        let y1 = (y0 + CHUNK_SIZE).min(gh - 1);
        let vx_n = x1 - x0 + 1;
        let vy_n = y1 - y0 + 1;
        let center_xz = self.grid_to_world_xz((x0 + 8) as f32, (y0 + 8) as f32);

        let mut cache = ChunkCache {
            center_xz,
            terrain_verts: Vec::with_capacity(vx_n * vy_n + 64),
            terrain_idxs: Vec::with_capacity((vx_n - 1) * (vy_n - 1) * 6 + 128),
            props: Vec::with_capacity(64),
            surfels: Vec::with_capacity(128),
            volumetrics: Vec::with_capacity(64),
            fire_hotspots: Vec::with_capacity(4),
            waterfall_points: Vec::new(),
        };

        // 1. Smoothed terrain grid
        for ly in 0..vy_n {
            let gy = y0 + ly;
            for lx in 0..vx_n {
                let gx = x0 + lx;
                let idx = gy * gw + gx;
                let c = &self.decoded[idx];
                let (_raw_h, rgb, _is_water, burn_t) = cell_height_and_color(c, gx, gy, gw, gh);
                let h = self.height_grid[idx];
                let (wx, wz) = self.grid_to_world_xz(gx as f32, gy as f32);
                let n = self.smooth_normals[idx];
                // Keep terrain burn_t moderate (<= 0.55) so soft Vol-3DGS flames shine without white clipping!
                cache.terrain_verts.push(MeshVertex {
                    pos: [wx, h, wz],
                    normal: n,
                    color_mat: [rgb[0], rgb[1], rgb[2], 9.0],
                    extra: [c.temp_norm, burn_t * 0.52, c.flow_x, c.flow_z],
                });
            }
        }

        for ly in 0..(vy_n - 1) {
            for lx in 0..(vx_n - 1) {
                let i00 = (ly * vx_n + lx) as u32;
                let i10 = i00 + 1;
                let i01 = i00 + vx_n as u32;
                let i11 = i01 + 1;
                cache
                    .terrain_idxs
                    .extend_from_slice(&[i00, i01, i11, i00, i11, i10]);
            }
        }

        let mut fire_sum = Vec3::ZERO;
        let mut fire_cnt = 0.0f32;

        // 2. Flowing water quads + Hard-Z Foliage Surfels + Soft Volumetric 3DGS Fire/Smoke/Steam!
        for dy in 0..CHUNK_SIZE {
            let gy = y0 + dy;
            if gy >= gh {
                continue;
            }
            for dx in 0..CHUNK_SIZE {
                let gx = x0 + dx;
                if gx >= gw {
                    continue;
                }
                let idx = gy * gw + gx;
                let c = &self.decoded[idx];
                let (_raw_h, _, is_water, burn_t) = cell_height_and_color(c, gx, gy, gw, gh);
                let h = self.height_grid[idx];
                let (wx, wz) = self.grid_to_world_xz(gx as f32 + 0.5, gy as f32 + 0.5);

                if is_water && gx + 1 < gw && gy + 1 < gh {
                    let (wx0, wz0) = self.grid_to_world_xz(gx as f32, gy as f32);
                    let wy = 0.14;
                    let base = cache.terrain_verts.len() as u32;
                    let extra = [c.temp_norm, 0.0, c.flow_x, c.flow_z];
                    for (px, pz) in [
                        (wx0, wz0),
                        (wx0, wz0 + 1.0),
                        (wx0 + 1.0, wz0 + 1.0),
                        (wx0 + 1.0, wz0),
                    ] {
                        cache.terrain_verts.push(MeshVertex {
                            pos: [px, wy, pz],
                            normal: [0.0, 1.0, 0.0],
                            color_mat: [0.18, 0.52, 0.78, 3.0],
                            extra,
                        });
                    }
                    cache
                        .terrain_idxs
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }

                let j0 = hash2i(gx as i32 * 17 + 3, gy as i32 * 31 + 7);
                let j1 = hash2i(gx as i32 * 43 - 5, gy as i32 * 19 + 11);

                // TIER 3 HYBRID SOFT VOLUMETRIC 3DGS: Wildfires + Rising Smoke Plumes!
                if c.burning {
                    let intensity = burn_t.max(0.85);
                    fire_sum += Vec3::new(wx, h + 1.5, wz) * intensity;
                    fire_cnt += intensity;
                    if j0 < 0.42 {
                        // Leaping translucent flame Gaussian
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(
                                wx + (j0 - 0.5) * 0.55,
                                h + 0.85 + j1 * 0.65,
                                wz + (j1 - 0.5) * 0.55,
                            ),
                            radius: 0.72 + j1 * 0.52,
                            height_stretch: 1.75 + j0 * 0.65,
                            color: [1.35, 0.48 + j1 * 0.24, 0.09],
                            alpha: 0.72,
                            kind: 0.0,
                            phase: j0 * 18.0 + j1 * 9.0,
                        });
                        // Billowing charcoal smoke Gaussian above the flame
                        if j1 < 0.55 {
                            cache.volumetrics.push(ChunkVolItem {
                                world_pos: Vec3::new(wx, h + 2.3 + j0 * 0.9, wz),
                                radius: 1.15 + j0 * 0.65,
                                height_stretch: 1.15,
                                color: [0.18, 0.16, 0.24],
                                alpha: 0.42,
                                kind: 1.0,
                                phase: j1 * 14.0,
                            });
                        }
                    }
                    if (c.s1 == SP_WOOD || c.s2 == SP_WOOD) && (gx % 3 == 0 && gy % 3 == 0) {
                        let sc = 1.85 + j1 * 0.80;
                        cache.props.push(ChunkPropItem {
                            kind: MeshKind::TreeTrunk,
                            world_pos: Vec3::new(wx + (j0 - 0.5) * 1.1, h - 0.14, wz + (j1 - 0.5) * 1.1),
                            yaw: j0 * 6.28,
                            scale: Vec3::new(sc * 0.90, sc * 1.05, sc * 0.90),
                            tint: [0.30, 0.18, 0.15],
                            glow: 0.18,
                        });
                    }
                    continue;
                } else if burn_t > 0.58 && !is_water {
                    fire_sum += Vec3::new(wx, h + 1.1, wz) * burn_t * 0.5;
                    fire_cnt += burn_t * 0.5;
                    if j0 < 0.14 {
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(wx, h + 0.65, wz),
                            radius: 0.60 + j1 * 0.35,
                            height_stretch: 1.45,
                            color: [1.20, 0.42, 0.10],
                            alpha: 0.52,
                            kind: 0.0,
                            phase: j1 * 12.0,
                        });
                    }
                }

                // TIER 3 HYBRID SOFT VOLUMETRIC 3DGS: Scalding Steam & Smoke Clouds!
                if (c.s1 == SP_STEAM || (c.s2 == SP_STEAM && c.s2_frac > 0.28)) && j0 < 0.38 {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, h + 1.2 + j1 * 0.8, wz),
                        radius: 1.35 + j1 * 0.70,
                        height_stretch: 1.10,
                        color: [0.86, 0.93, 0.98],
                        alpha: 0.52,
                        kind: 2.0,
                        phase: j0 * 15.0,
                    });
                } else if (c.s1 == SP_SMOKE || (c.s2 == SP_SMOKE && c.s2_frac > 0.30)) && j0 < 0.24 {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, h + 1.6 + j1 * 0.8, wz),
                        radius: 1.10 + j0 * 0.55,
                        height_stretch: 1.05,
                        color: [0.22, 0.20, 0.29],
                        alpha: 0.38,
                        kind: 1.0,
                        phase: j1 * 11.0,
                    });
                }

                // APPROACH 4 HARD-Z SDF FOLIAGE SURFELS: Majestic Towering Nordic Pines (~3x-4x hero height!)
                // Spaced organically (1 tall pine per 3x3 forest cluster + mossy undergrowth) so heroes weave between trunks!
                let is_bridge = c.s1 == SP_WOOD && !c.wall && c.walkable;
                if !is_bridge && (c.s1 == SP_WOOD || (c.s2 == SP_WOOD && c.s2_frac > 0.35)) {
                    let snowy = c.freezing || c.temp_norm < 0.088 || (gy as f32 / gh as f32) < 0.32;
                    if gx % 3 == 0 && gy % 3 == 0 {
                        let scale = 1.95 + j0 * 0.95; // ~9-13.5u tall towering pine!
                        let jx = (j0 - 0.5) * 1.45;
                        let jz = (j1 - 0.5) * 1.45;
                        cache.props.push(ChunkPropItem {
                            kind: MeshKind::TreeTrunk,
                            world_pos: Vec3::new(wx + jx, h - 0.14, wz + jz),
                            yaw: j0 * 6.28,
                            scale: Vec3::new(scale * 0.92, scale * 1.08, scale * 0.92),
                            tint: [1.0, 1.0, 1.0],
                            glow: 0.0,
                        });
                        let tint = if c.temp_norm > 0.16 {
                            [1.05, 0.82, 0.56]
                        } else if snowy {
                            [0.95, 0.99, 1.02]
                        } else {
                            [0.84 + j1 * 0.24, 0.90 + j0 * 0.16, 0.86 + j1 * 0.14]
                        };
                        cache.surfels.push(ChunkSurfelItem {
                            kind: if snowy {
                                SurfelProtoKind::SnowPine
                            } else {
                                SurfelProtoKind::PineCanopy
                            },
                            world_pos: Vec3::new(wx + jx, h, wz + jz),
                            yaw: j1 * 6.28,
                            scale,
                            tint,
                            wind_phase: gx as f32 * 0.37 + gy as f32 * 0.51,
                            glow_mul: 0.0,
                        });
                    } else if j0 < 0.10 {
                        // Waist-high forest floor undergrowth & saplings between the giant trunks
                        cache.surfels.push(ChunkSurfelItem {
                            kind: SurfelProtoKind::MarshBrush,
                            world_pos: Vec3::new(wx + (j0 - 0.5) * 0.5, h, wz + (j1 - 0.5) * 0.5),
                            yaw: j1 * 6.28,
                            scale: 0.85 + j1 * 0.45,
                            tint: if snowy { [0.74, 0.86, 0.94] } else { [0.82, 0.94, 0.84] },
                            wind_phase: j0 * 9.0,
                            glow_mul: 0.08,
                        });
                    }
                } else if (c.s1 == SP_PLANT || (c.s2 == SP_PLANT && c.s2_frac > 0.28)) && j0 < 0.14 {
                    let near_water = (gx as i32 - (gw as i32 / 2)).abs() < (gw as i32 / 5);
                    cache.surfels.push(ChunkSurfelItem {
                        kind: if near_water && j1 < 0.55 {
                            SurfelProtoKind::RiverReeds
                        } else {
                            SurfelProtoKind::MarshBrush
                        },
                        world_pos: Vec3::new(wx + (j0 - 0.5) * 0.5, h, wz + (j1 - 0.5) * 0.5),
                        yaw: j1 * 6.28,
                        scale: 0.92 + j0 * 0.52,
                        tint: [0.90 + j1 * 0.18, 0.96, 0.88 + j0 * 0.16],
                        wind_phase: j0 * 10.0,
                        glow_mul: 0.15,
                    });
                } else if c.s1 == SP_ROCK
                    && gx >= 5
                    && gx + 5 < gw
                    && gy >= 5
                    && gy + 5 < gh
                    && gx % 4 == 0
                    && gy % 4 == 0
                {
                    cache.props.push(ChunkPropItem {
                        kind: MeshKind::RockOutcrop,
                        world_pos: Vec3::new(wx, h - 1.15, wz),
                        yaw: j0 * 6.28,
                        scale: Vec3::new(1.75 + j0 * 0.75, 1.85 + j1 * 0.95, 1.75 + j1 * 0.75),
                        tint: [1.0, 1.0, 1.0],
                        glow: 0.0,
                    });
                }

                // LIVE ECOSYSTEM WILDLIFE HERDS + Soft Bioluminescent Marsh Wisps (natural meadow/forest herds!)
                if c.critter >= 2
                    && c.walkable
                    && !c.burning
                    && c.s1 != SP_OIL
                    && c.s1 != SP_WATER
                    && (gx % 8 == 0 && gy % 8 == 0)
                {
                    if j0 < 0.42 {
                        cache.props.push(ChunkPropItem {
                            kind: MeshKind::MarshDeer,
                            world_pos: Vec3::new(wx + (j0 - 0.5) * 1.4, h, wz + (j1 - 0.5) * 1.4),
                            yaw: j1 * 6.28,
                            scale: Vec3::splat(0.95 + j0 * 0.24),
                            tint: [1.0, 1.0, 1.0],
                            glow: 0.0,
                        });
                    }
                    if c.critter >= 4 && j1 < 0.45 {
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(wx, h + 1.4, wz),
                            radius: 0.55,
                            height_stretch: 1.0,
                            color: [0.52, 1.15, 0.65],
                            alpha: 0.58,
                            kind: 3.0,
                            phase: j0 * 12.0,
                        });
                    }
                }
            }
        }

        if fire_cnt > 0.0 {
            cache.fire_hotspots.push((fire_sum / fire_cnt, fire_cnt));
        }
        self.chunks[ci] = cache;
    }

    fn recompute_summary(&mut self) {
        let mut s = WorldSummaryStats::default();
        for c in &self.decoded {
            let has = |sp: u8| c.s1 == sp || (c.s2 == sp && c.s2_frac > 0.25);
            if has(SP_WATER) { s.water_cells += 1; }
            if has(SP_ICE) { s.ice_cells += 1; }
            if has(SP_STEAM) { s.steam_cells += 1; }
            if has(SP_ROCK) { s.rock_cells += 1; }
            if has(SP_WOOD) { s.wood_cells += 1; }
            if has(SP_ASH) { s.ash_cells += 1; }
            if has(SP_OIL) { s.oil_cells += 1; }
            if has(SP_PLANT) { s.plant_cells += 1; }
            if c.burning { s.burning_cells += 1; }
            if c.freezing { s.freezing_cells += 1; }
            if c.critter >= 2 { s.critter_cells += 1; }
        }
        for ch in &self.chunks {
            for sf in &ch.surfels {
                match sf.kind {
                    SurfelProtoKind::PineCanopy | SurfelProtoKind::SnowPine => {
                        s.total_pine_trees += 1
                    }
                    SurfelProtoKind::MarshBrush | SurfelProtoKind::RiverReeds => {
                        s.total_brush_clusters += 1
                    }
                    _ => {}
                }
            }
            s.total_volumetric_splats += ch.volumetrics.len();
            for p in &ch.props {
                if p.kind == MeshKind::MarshDeer {
                    s.total_wildlife_deer += 1;
                }
            }
        }
        self.summary = s;
    }

    pub fn sample_ground_y(&self, gx: f32, gy: f32) -> f32 {
        let xi = (gx.floor() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
        let yi = (gy.floor() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
        self.height_grid[yi * self.grid_w + xi].max(0.08)
    }

    /// Samples the nearest 3D waterfall centroid and nearby weighted wildfire centroid for spatial audio.
    /// Returns `(nearest_waterfall: Option<(world_pos, dist)>, nearby_fire: Option<(world_pos, intensity)>)`.
    pub fn sample_ambient_audio_sources(
        &self,
        listener_pos: Vec3,
    ) -> (Option<(Vec3, f32)>, Option<(Vec3, f32)>) {
        let mut best_wf: Option<(Vec3, f32)> = None;
        for ch in &self.chunks_l2 {
            for &wp in &ch.waterfall_points {
                let d = wp.distance(listener_pos);
                if best_wf.map_or(true, |(_, bd)| d < bd) {
                    best_wf = Some((wp, d));
                }
            }
        }

        let mut fire_pos_sum = Vec3::ZERO;
        let mut fire_weight_sum = 0.0f32;
        for ch in self.chunks.iter().chain(self.chunks_l2.iter()) {
            for &(fp, inten) in &ch.fire_hotspots {
                let d = fp.distance(listener_pos);
                if d < 72.0 {
                    let w = inten / (1.0 + (d / 18.0).powi(2));
                    fire_pos_sum += fp * w;
                    fire_weight_sum += w;
                }
            }
        }
        let nearby_fire = if fire_weight_sum > 0.05 {
            Some((fire_pos_sum / fire_weight_sum, fire_weight_sum))
        } else {
            None
        };
        (best_wf, nearby_fire)
    }

    /// Map authoritative 3D layer altitude `gz` (`1.0=ground`, `2.0=Highland/Skybridge`) into world Y!
    pub fn sample_3d_y(&self, gx: f32, gy: f32, gz: f32) -> f32 {
        let base_y = self.sample_ground_y(gx, gy);
        let surf_z = self.surf_layer() as f32;
        if gz <= surf_z + 0.02 {
            base_y
        } else {
            let t = (gz - surf_z).clamp(0.0, 2.0);
            // Smoothly interpolates from surface up onto monumental L2 plateau/skybridge deck (y ≈ 8.25)
            let l2_target = 8.25f32.max(base_y + 4.5);
            if t <= 1.0 {
                base_y + (l2_target - base_y) * t
            } else {
                l2_target + (t - 1.0) * 6.5
            }
        }
    }
}

pub struct AssembledFrame {
    pub terrain_verts: Vec<MeshVertex>,
    pub terrain_idxs: Vec<u32>,
    pub mesh_batches: [Vec<MeshInstance>; MESH_KIND_COUNT],
    pub surfel_placements: Vec<SplatPlacement>,
    pub surfel_refs: Vec<SplatRef>,
    pub vol_splats: Vec<VolSplat>,
    pub point_pos: [[f32; 4]; 8],
    pub point_col: [[f32; 4]; 8],
    pub visible_chunks: usize,
    #[allow(dead_code)]
    pub squad_arc_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AimReticle {
    pub gx: f32,
    pub gy: f32,
    pub gz: f32,
    pub radius: f32,
    pub color: [f32; 3],
}

/// Exact local-to-world-offset transform matching `mesh_pass.wgsl` & `shadow_pass.wgsl`
/// (scale -> roll around Z -> pitch around X -> yaw around Y) so articulated Skywing wings
/// stay rigidly locked to shoulder sockets while pitching and banking!
#[inline]
fn transform_local_offset(local: Vec3, scale: Vec3, roll: f32, pitch: f32, yaw: f32) -> Vec3 {
    let s = local * scale;
    let (cr, sr) = (roll.cos(), roll.sin());
    let r = Vec3::new(s.x * cr - s.y * sr, s.x * sr + s.y * cr, s.z);
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let p = Vec3::new(r.x, r.y * cp - r.z * sp, r.y * sp + r.z * cp);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    Vec3::new(p.x * cy - p.z * sy, p.y, p.x * sy + p.z * cy)
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_live_frame(
    world: &LiveApproach4World,
    z16: Option<&Z16World>,
    entities: Option<&LiveEntityState>,
    surfels_bank: &SurfelProtoBank,
    mats: &CameraMatrices,
    cam_target: Vec3,
    cam_dist: f32,
    sim_time: f32,
    layer_slice_mode: u8, // 0=ALL 4L (3D), 1=GROUND (L0-L1), 2=HIGH+SKY (L2-L3)
    aim_reticle: Option<AimReticle>,
    show_squad_links: bool,
) -> AssembledFrame {
    let origin = mats.origin;
    let z_depth = z16.map(|z| z.depth).unwrap_or(world.grid_d);
    let cull_rad = (cam_dist * 1.95 + if z_depth >= 48 { 88.0 } else { 42.0 }).max(if z_depth >= 48 { 160.0 } else { 96.0 });
    let cull_r2 = cull_rad * cull_rad;

    let mut terrain_verts = Vec::with_capacity(if z16.is_some() { 0 } else { 48_000 });
    let mut terrain_idxs = Vec::with_capacity(if z16.is_some() { 0 } else { 96_000 });
    let mut mesh_batches: [Vec<MeshInstance>; MESH_KIND_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut surfel_placements = Vec::with_capacity(3_072);
    let mut surfel_refs = Vec::with_capacity(80_000);
    let mut vol_splats = Vec::with_capacity(2_048);

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
    let sample_shrine_y = |gx: f32, gy: f32, name: &str| -> f32 {
        if let Some(z16w) = z16 {
            if z16w.depth >= 48 {
                if name.contains("ABYSSAL") {
                    return z16w.sample_surface_y_near(gx, gy, 10.0);
                } else if name.contains("VIADUCT") {
                    return z16w.sample_surface_y_near(gx, gy, 35.0);
                }
            }
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

    let mut add_surfel = |kind: SurfelProtoKind,
                          world_pos: Vec3,
                          yaw: f32,
                          scale: f32,
                          tint: [f32; 3],
                          wind_phase: f32,
                          glow_mul: f32| {
        let rel = world_pos - origin;
        let dist = (rel - mats.rel_eye).length();
        let (stride, rad_mul) = if dist > 125.0 {
            (4usize, 1.65f32)
        } else if dist > 86.0 {
            (2usize, 1.28f32)
        } else {
            (1usize, 1.0f32)
        };
        let p_idx = surfel_placements.len() as u32;
        surfel_placements.push(SplatPlacement {
            pos_yaw: [rel.x, rel.y, rel.z, yaw],
            tint_scale: [tint[0], tint[1], tint[2], scale],
            lod_phase: [rad_mul, wind_phase, glow_mul, 0.0],
        });
        let r = surfels_bank.range(kind);
        let mut k = 0u32;
        while k < r.count {
            surfel_refs.push(SplatRef {
                placement_idx: p_idx,
                proto_idx: r.start + k,
            });
            k += stride as u32;
        }
    };

    let mut all_fires: Vec<(Vec3, f32)> = Vec::new();
    let mut visible_chunks = 0usize;

    let mut emit_chunk = |ch: &ChunkCache| {
        if ch.terrain_idxs.is_empty() && ch.props.is_empty() && ch.surfels.is_empty() && ch.volumetrics.is_empty() {
            return;
        }
        let dx = ch.center_xz.0 - cam_target.x;
        let dz = ch.center_xz.1 - cam_target.z;
        if dx * dx + dz * dz > cull_r2 {
            return;
        }
        visible_chunks += 1;

        let base_v = terrain_verts.len() as u32;
        for v in &ch.terrain_verts {
            let mut rv = *v;
            rv.pos[0] -= origin.x;
            rv.pos[1] -= origin.y;
            rv.pos[2] -= origin.z;
            terrain_verts.push(rv);
        }
        for &idx in &ch.terrain_idxs {
            terrain_idxs.push(base_v + idx);
        }

        for p in &ch.props {
            let rel = p.world_pos - origin;
            mesh_batches[p.kind as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, p.yaw],
                scale_pitch: [p.scale.x, p.scale.y, p.scale.z, 0.0],
                tint_glow: [p.tint[0], p.tint[1], p.tint[2], p.glow],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
        }

        for sf in &ch.surfels {
            add_surfel(sf.kind, sf.world_pos, sf.yaw, sf.scale, sf.tint, sf.wind_phase, sf.glow_mul);
        }

        for vl in &ch.volumetrics {
            let rel = vl.world_pos - origin;
            vol_splats.push(VolSplat {
                pos_rad: [rel.x, rel.y, rel.z, vl.radius],
                color_alpha: [vl.color[0], vl.color[1], vl.color[2], vl.alpha],
                params: [vl.height_stretch, vl.kind, vl.phase, 1.0],
            });
        }

        all_fires.extend_from_slice(&ch.fire_hotspots);
    };

    if let Some(z16w) = z16 {
        // True 3D Z=16 mode: Surface-Nets renders terrain directly; emit Hard-Z sculpted pines & fire hotspots!
        visible_chunks = z16w.non_empty_bricks();
        for pines in &z16w.brick_pines {
            for p in pines {
                let dx = p.world_pos.x - cam_target.x;
                let dz = p.world_pos.z - cam_target.z;
                if dx * dx + dz * dz > cull_r2 {
                    continue;
                }
                let kind = if p.snowy {
                    SurfelProtoKind::SnowPine
                } else {
                    SurfelProtoKind::PineCanopy
                };
                add_surfel(kind, p.world_pos, p.yaw, p.scale, p.tint, p.wind_phase, 0.0);
            }
        }
        for flist in &z16w.brick_fires {
            all_fires.extend_from_slice(flist);
        }
    } else {
        if layer_slice_mode != 2 {
            for ch in &world.chunks {
                emit_chunk(ch);
            }
        }
        if layer_slice_mode != 1 {
            for ch in &world.chunks_l2 {
                emit_chunk(ch);
            }
            for ch in &world.chunks_l3 {
                emit_chunk(ch);
            }
        }
    }

    // Dynamic 8-slot point lights (softly balanced to illuminate terrain without clipping white!)
    let mut point_pos = [[0.0f32; 4]; 8];
    let mut point_col = [[0.0f32; 4]; 8];
    let mut light_n = 0usize;

    let mut push_light = |wpos: Vec3, radius: f32, rgb: [f32; 3]| {
        if light_n < 8 {
            let rel = wpos - origin;
            point_pos[light_n] = [rel.x, rel.y, rel.z, radius];
            point_col[light_n] = [rgb[0], rgb[1], rgb[2], 1.0];
            light_n += 1;
        }
    };

    all_fires.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for &(fpos, inten) in all_fires.iter().take(4) {
        let flicker = 0.88 + 0.14 * (sim_time * 8.5 + fpos.x * 0.7).sin();
        let rad = (10.0 + inten.sqrt() * 2.0).min(20.0);
        push_light(fpos, rad, [0.95 * flicker, 0.44 * flicker, 0.14 * flicker]);
    }

    let mut squad_arc_count = 0usize;

    // 30 Hz Entities: Shrines, Rafts, Golden Relic, 5-Archetype Heroes, Squad Links, Projectiles!
    if let Some(ent) = entities {
        // 3 Capturable Elemental Shrines
        for s in &ent.shrines {
            let (wx, wz) = grid_to_xz(s.x, s.y);
            let gy = sample_shrine_y(s.x, s.y, &s.name);
            let rel = Vec3::new(wx, gy - 0.05, wz) - origin;
            let scol = if s.progress < -0.15 {
                let t = (-s.progress).clamp(0.0, 1.0);
                [0.85 * (1.0 - t) + 0.24 * t, 0.80 * (1.0 - t) + 0.72 * t, 0.55 * (1.0 - t) + 1.0 * t]
            } else if s.progress > 0.15 {
                let t = s.progress.clamp(0.0, 1.0);
                [0.85 * (1.0 - t) + 1.0 * t, 0.80 * (1.0 - t) + 0.34 * t, 0.55 * (1.0 - t) + 0.22 * t]
            } else {
                [0.98, 0.86, 0.44]
            };
            let shrine_scale = if let Some(z16w) = z16 {
                (z16w.width as f32 / 192.0).clamp(0.68, 1.0) * 1.52
            } else {
                1.72
            };
            let pulse = 0.88 + 0.12 * (sim_time * 2.8 + s.id as f32 * 1.7).sin();
            mesh_batches[MeshKind::ShrineObelisk as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, sim_time * 0.35 + s.id as f32],
                scale_pitch: [shrine_scale, shrine_scale, shrine_scale, 0.0],
                tint_glow: [scol[0], scol[1], scol[2], 0.42],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            // Sleek double counter-rotating animated runic energy rings tightly hugging the shrine pedestal!
            // Open runic gaps discard cleanly so the stone viaduct arch beneath shines right through!
            let inner_r = shrine_scale * 2.18;
            let outer_r = shrine_scale * 3.05;
            mesh_batches[MeshKind::ShrineRuneRing as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y + 0.10, rel.z, sim_time * 0.55 + s.id as f32],
                scale_pitch: [inner_r, 0.90, inner_r, 0.0],
                tint_glow: [scol[0] * pulse, scol[1] * pulse, scol[2] * pulse, 0.88],
                flags: [-1.0, 0.0, 0.0, 12.0],
            });
            mesh_batches[MeshKind::ShrineRuneRing as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y + 0.13, rel.z, -sim_time * 0.36 + s.id as f32 * 2.1],
                scale_pitch: [outer_r, 0.78, outer_r, 0.0],
                tint_glow: [scol[0] * 0.92 * pulse, scol[1] * 0.92 * pulse, scol[2] * 0.92 * pulse, 0.72],
                flags: [-1.0, 0.0, 0.0, 18.0],
            });
            // Delicate tilted runic halo orbiting the floating crystal monolith!
            let crystal_y = rel.y + 4.35 * shrine_scale;
            mesh_batches[MeshKind::ShrineRuneRing as usize].push(MeshInstance {
                pos_yaw: [rel.x, crystal_y, rel.z, sim_time * 1.15 + s.id as f32],
                scale_pitch: [shrine_scale * 1.25, 0.70, shrine_scale * 1.25, 0.32],
                tint_glow: [
                    (scol[0] * 0.8 + 0.25) * pulse,
                    (scol[1] * 0.8 + 0.25) * pulse,
                    (scol[2] * 0.8 + 0.25) * pulse,
                    0.95,
                ],
                flags: [-1.0, 0.0, 0.18, 10.0],
            });
            // Soft translucent additive-feel ground light & floating crystal monolith aura!
            vol_splats.push(VolSplat {
                pos_rad: [rel.x, rel.y + 0.45, rel.z, outer_r * 0.85],
                color_alpha: [scol[0] * 0.95, scol[1] * 0.95, scol[2] * 0.95, 0.36 * pulse],
                params: [0.45, 6.0, s.id as f32 * 2.7, 0.0],
            });
            vol_splats.push(VolSplat {
                pos_rad: [rel.x, crystal_y + 0.35, rel.z, 1.55 * shrine_scale],
                color_alpha: [scol[0] * 1.15, scol[1] * 1.15, scol[2] * 1.15, 0.56 * pulse],
                params: [1.55, 3.0, s.id as f32 * 3.1, 0.0],
            });
            push_light(Vec3::new(wx, gy + 5.8, wz), 22.0, [scol[0] * 1.05, scol[1] * 1.05, scol[2] * 1.05]);
        }

        // Carved `CellBody` Ice Rafts
        for raft in &ent.bodies {
            let (wx, wz) = grid_to_xz(raft.x, raft.y);
            let bob = 0.06 * (sim_time * 3.1 + raft.id as f32).sin();
            let base_y = if z16.is_some() { raft.z.max(1.8) } else { 0.04 };
            let rel = Vec3::new(wx, base_y + bob, wz) - origin;
            let rad_s = raft.radius.clamp(1.6, 5.5);
            mesh_batches[MeshKind::IceRaft as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, (raft.id as f32) * 0.9 + sim_time * 0.18],
                scale_pitch: [rad_s, rad_s * 0.75, rad_s, 0.0],
                tint_glow: [1.0, 1.0, 1.0, 0.12],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            vol_splats.push(VolSplat {
                pos_rad: [rel.x, rel.y + 0.55, rel.z, 1.2],
                color_alpha: [0.58, 0.92, 1.0, 0.45],
                params: [0.8, 3.0, raft.id as f32, 0.0],
            });
        }

        // Golden Relic + Pedestal
        let any_carrier = ent.players.iter().any(|p| p.carrying_relic);
        let (rx, rz) = grid_to_xz(ent.relic_x, ent.relic_y);
        let ry = sample_g_y(ent.relic_x, ent.relic_y);
        if !any_carrier {
            let rel_ped = Vec3::new(rx, ry - 0.05, rz) - origin;
            mesh_batches[MeshKind::RelicPedestal as usize].push(MeshInstance {
                pos_yaw: [rel_ped.x, rel_ped.y, rel_ped.z, 0.0],
                scale_pitch: [1.0, 1.0, 1.0, 0.0],
                tint_glow: [1.0, 1.0, 1.0, 0.0],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            let relic_y = ry + 2.4 + 0.25 * (sim_time * 3.2).sin();
            let rel_r = Vec3::new(rx, relic_y, rz) - origin;
            mesh_batches[MeshKind::GoldenRelic as usize].push(MeshInstance {
                pos_yaw: [rel_r.x, rel_r.y, rel_r.z, sim_time * 1.4],
                scale_pitch: [1.1, 1.1, 1.1, 0.0],
                tint_glow: [1.0, 0.95, 0.78, 0.28],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            vol_splats.push(VolSplat {
                pos_rad: [rel_r.x, rel_r.y + 0.8, rel_r.z, 1.35],
                color_alpha: [1.0, 0.86, 0.32, 0.65],
                params: [1.2, 3.0, 2.0, 0.0],
            });
        }

        // 5-Archetype Heroes & Bots (scales smoothly to 40–64 live players in true 3D `(x, y, z)`!)
        for p in &ent.players {
            let (wx, wz) = grid_to_xz(p.x, p.y);
            let ground_y = sample_g_y(p.x, p.y);
            let gy = sample_e_y(p.x, p.y, p.z);
            let altitude_above_ground = (gy - ground_y).max(0.0);
            let speed = p.vx.hypot(p.vy);
            let phase = p.id as f32 * 1.73;

            // Local hero meshes face +Z; mesh_pass rotates +Z to (-sin(yaw), cos(yaw)),
            // so yaw = (-aim_x).atan2(aim_y) aligns facing exactly with (aim_x, aim_y)!
            let yaw = (-p.aim_x).atan2(p.aim_y);
            let aim_len = p.aim_x.hypot(p.aim_y).max(0.001);
            let lat_vel = (p.aim_x * p.vy - p.aim_y * p.vx) / aim_len;

            let team_tint = if p.team == 0 {
                [0.20, 0.58, 0.98]
            } else {
                [0.94, 0.24, 0.26]
            };
            let is_self = p.id == ent.your_player_id;
            let ring_col = if is_self { [0.38, 1.0, 0.56] } else { team_tint };
            let base_glow = if p.dashing || p.ult_active { 0.30 } else { 0.0 };

            let (kind, body_scale, bob, move_pitch, body_roll) = match p.class_id {
                CLASS_STRIDER => {
                    // Heavy Walker / Squad Anchor: sturdy grounded rhythmic stride + slight shoulder sway
                    let stride = if speed > 0.8 {
                        (sim_time * 7.8 + phase).sin().abs() * 0.15
                    } else {
                        0.0
                    };
                    let sway = if speed > 0.8 {
                        (sim_time * 7.8 + phase).sin() * 0.055
                    } else {
                        0.0
                    };
                    let pitch = if p.dashing {
                        0.20
                    } else if speed > 1.0 {
                        0.08
                    } else {
                        0.0
                    };
                    (MeshKind::HeroStrider, Vec3::new(1.18, 1.16, 1.18), stride, pitch, sway)
                }
                CLASS_SKYWING => {
                    // True 3D Flyer / Thermal Glider: rhythmic airborne hover/flapping + banking glide!
                    let airborne = altitude_above_ground > 0.55
                        || p.vz.abs() > 0.35
                        || (z16.is_none() && p.z > 1.35);
                    let gliding = airborne && speed > 11.5 && p.vz <= 0.45;
                    let bob = if airborne {
                        let freq = if gliding { 3.4 } else { 9.6 };
                        0.24 - 0.16 * (sim_time * freq + phase).cos()
                    } else if speed > 0.9 {
                        (sim_time * 10.5 + phase).sin().abs() * 0.18
                    } else {
                        0.0
                    };
                    let bank = if airborne {
                        (-lat_vel * 0.032
                            + (sim_time * 1.8 + phase).sin() * if gliding { 0.08 } else { 0.03 })
                        .clamp(-0.42, 0.42)
                    } else {
                        (-lat_vel * 0.012).clamp(-0.14, 0.14)
                    };
                    let pitch = if p.dashing {
                        0.36
                    } else if gliding {
                        0.28
                    } else if airborne {
                        (0.14 - p.vz * 0.045).clamp(-0.18, 0.32)
                    } else if speed > 1.0 {
                        0.09
                    } else {
                        0.0
                    };
                    (MeshKind::HeroSkywing, Vec3::new(1.08, 1.12, 1.08), bob, pitch, bank)
                }
                CLASS_TIDEKIN => {
                    // Amphibious Swimmer / Diver: fluid hydrodynamic bob + lateral swimming undulation
                    let bob = if speed > 0.8 {
                        (sim_time * 9.5 + phase).sin() * 0.12 + 0.08
                    } else {
                        (sim_time * 3.0 + phase).sin() * 0.05
                    };
                    let swim_roll = (sim_time * if speed > 0.8 { 7.5 } else { 2.6 } + phase).sin()
                        * if speed > 0.8 { 0.13 } else { 0.04 }
                        + (-lat_vel * 0.018).clamp(-0.18, 0.18);
                    let pitch = if p.dashing {
                        0.38
                    } else if speed > 1.0 {
                        0.18
                    } else {
                        0.04
                    };
                    (MeshKind::HeroTidekin, Vec3::new(1.08, 1.12, 1.12), bob, pitch, swim_roll)
                }
                CLASS_BURROWER => {
                    // Subterranean Crystal-Plated Golem: rhythmic drilling lunge & heavy lateral sway!
                    let active_drill = speed > 0.8 || p.melee_active || p.dashing;
                    let drill_freq = if p.melee_active || p.dashing { 15.5 } else { 10.2 };
                    let drill_osc = (sim_time * drill_freq + phase).sin();
                    let bob = if active_drill {
                        drill_osc.abs() * 0.14 + 0.04
                    } else {
                        (sim_time * 3.2 + phase).sin() * 0.035
                    };
                    let drill_sway = if active_drill {
                        drill_osc * 0.12
                    } else {
                        (sim_time * 2.4 + phase).sin() * 0.03
                    } + (-lat_vel * 0.015).clamp(-0.14, 0.14);
                    let pitch = if p.dashing {
                        0.36
                    } else if p.melee_active {
                        0.28 + drill_osc * 0.10
                    } else if speed > 0.8 {
                        0.16 + drill_osc.abs() * 0.08
                    } else {
                        0.05
                    };
                    (MeshKind::HeroBurrower, Vec3::new(1.16, 1.08, 1.18), bob, pitch, drill_sway)
                }
                _ => {
                    // CLASS_EMBERSTALKER (3) — Solo Cavern Climber / Raider: agile low-slung crouched gait
                    let bob = if speed > 0.8 {
                        (sim_time * 13.5 + phase).sin().abs() * 0.15
                    } else {
                        (sim_time * 2.4 + phase).sin() * 0.03
                    };
                    let lean = (-lat_vel * 0.025).clamp(-0.22, 0.22);
                    let pitch = if p.dashing {
                        0.36
                    } else if speed > 1.0 {
                        0.24
                    } else {
                        0.10
                    };
                    (MeshKind::HeroStalker, Vec3::new(1.04, 1.06, 1.06), bob, pitch, lean)
                }
            };

            let rel = Vec3::new(wx, gy + bob, wz) - origin;
            mesh_batches[kind as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, yaw],
                scale_pitch: [body_scale.x, body_scale.y, body_scale.z, move_pitch],
                tint_glow: [team_tint[0], team_tint[1], team_tint[2], base_glow],
                flags: [-1.0, 1.0, body_roll, 0.0],
            });

            // Archetype-specific articulated attachments, trails & Squad/Solo visual indicators!
            match p.class_id {
                CLASS_STRIDER => {
                    // Crisp inner footprint + outer Squad Anchor rally halo ring!
                    let ring_rel = Vec3::new(wx, gy + 0.05, wz) - origin;
                    let inner_s = if is_self { 1.28 } else { 1.08 };
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y, ring_rel.z, 0.0],
                        scale_pitch: [inner_s, 1.0, inner_s, 0.0],
                        tint_glow: [ring_col[0], ring_col[1], ring_col[2], 0.78],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                    let anchor_pulse = 1.62 + 0.08 * (sim_time * 2.4 + phase).sin();
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y + 0.02, ring_rel.z, sim_time * 0.55 + phase],
                        scale_pitch: [anchor_pulse, 0.72, anchor_pulse, 0.0],
                        tint_glow: [
                            team_tint[0] * 0.75 + 0.25,
                            team_tint[1] * 0.75 + 0.25,
                            team_tint[2] * 0.75 + 0.25,
                            0.36,
                        ],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                }
                CLASS_SKYWING => {
                    let airborne = altitude_above_ground > 0.55
                        || p.vz.abs() > 0.35
                        || (z16.is_none() && p.z > 1.35);
                    let gliding = airborne && speed > 11.5 && p.vz <= 0.45;
                    let soaring_high =
                        altitude_above_ground > 1.25 || (z16.is_none() && p.z > 1.45);
                    let flap_rad = if gliding {
                        0.18 + 0.14 * (sim_time * 3.4 + phase).sin()
                    } else if airborne {
                        0.20 + 0.62 * (sim_time * 9.6 + phase).sin()
                    } else {
                        -0.28 + 0.07 * (sim_time * 2.8 + phase).sin()
                    };
                    let wing_pitch = move_pitch + if gliding { 0.08 } else { 0.0 };
                    let wing_scale = if gliding {
                        Vec3::new(1.16, 1.08, 1.12)
                    } else {
                        Vec3::new(1.08, 1.08, 1.08)
                    };
                    let wing_glow = if p.dashing || p.ult_active {
                        0.45
                    } else if airborne {
                        0.22
                    } else {
                        0.08
                    };

                    // Mount Left & Right glowing swept wings cleanly to the Skywing's back shoulder sockets
                    let off_l = transform_local_offset(
                        Vec3::new(-0.28, 1.56, -0.24),
                        body_scale,
                        body_roll,
                        move_pitch,
                        yaw,
                    );
                    let off_r = transform_local_offset(
                        Vec3::new(0.28, 1.56, -0.24),
                        body_scale,
                        body_roll,
                        move_pitch,
                        yaw,
                    );
                    let wl_pos = rel + off_l;
                    let wr_pos = rel + off_r;
                    mesh_batches[MeshKind::SkywingWingL as usize].push(MeshInstance {
                        pos_yaw: [wl_pos.x, wl_pos.y, wl_pos.z, yaw],
                        scale_pitch: [wing_scale.x, wing_scale.y, wing_scale.z, wing_pitch],
                        tint_glow: [team_tint[0], team_tint[1], team_tint[2], wing_glow],
                        flags: [-1.0, 1.0, body_roll - flap_rad, 0.0],
                    });
                    mesh_batches[MeshKind::SkywingWingR as usize].push(MeshInstance {
                        pos_yaw: [wr_pos.x, wr_pos.y, wr_pos.z, yaw],
                        scale_pitch: [wing_scale.x, wing_scale.y, wing_scale.z, wing_pitch],
                        tint_glow: [team_tint[0], team_tint[1], team_tint[2], wing_glow],
                        flags: [-1.0, 1.0, body_roll + flap_rad, 0.0],
                    });

                    // Ground projection ring beneath the flyer (always readable in 3D!)
                    let proj_y = if airborne { ground_y + 0.06 } else { gy + 0.05 };
                    let ring_rel = Vec3::new(wx, proj_y, wz) - origin;
                    let rs = if is_self { 1.24 } else { 1.00 };
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y, ring_rel.z, 0.0],
                        scale_pitch: [rs, 0.9, rs, 0.0],
                        tint_glow: [ring_col[0], ring_col[1], ring_col[2], if airborne { 0.56 } else { 0.72 }],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });

                    // Visible rising helical Thermal Updraft column under soaring Skywings!
                    if soaring_high || airborne {
                        let max_col_h = if z_depth >= 48 { 18.0 } else { 7.5 };
                        let col_h = (gy - ground_y).clamp(1.4, max_col_h);
                        for k in 0..3 {
                            let t_cyc = (sim_time * 1.30 + phase * 0.37 + k as f32 * 0.333).fract();
                            let th_y = ground_y + 0.35 + t_cyc * col_h;
                            let spiral_a = sim_time * 3.4 + phase + t_cyc * 5.0;
                            let spiral_r = (0.28 + (1.0 - t_cyc) * 0.32) * (k as f32 * 0.25 + 0.65);
                            let th_rel = Vec3::new(
                                wx + spiral_a.cos() * spiral_r - p.vx * 0.05 * (1.0 - t_cyc),
                                th_y,
                                wz + spiral_a.sin() * spiral_r - p.vy * 0.05 * (1.0 - t_cyc),
                            ) - origin;
                            let fade_up = (t_cyc * std::f32::consts::PI).sin().clamp(0.15, 1.0);
                            vol_splats.push(VolSplat {
                                pos_rad: [th_rel.x, th_rel.y, th_rel.z, 0.62 + (1.0 - t_cyc) * 0.42],
                                color_alpha: [
                                    0.82 + 0.32 * (1.0 - t_cyc),
                                    0.94,
                                    0.58 + 0.50 * t_cyc,
                                    0.44 * fade_up,
                                ],
                                params: [0.68, 6.0, phase + k as f32 * 2.1, 0.2],
                            });
                        }
                        let t_ring = (sim_time * 1.45 + phase * 0.37).fract();
                        let ring_y = ground_y + 0.45 + t_ring * col_h;
                        let ring_s = 0.85 + t_ring * 0.95;
                        let th_rel = Vec3::new(wx, ring_y, wz) - origin;
                        mesh_batches[MeshKind::ShrineRuneRing as usize].push(MeshInstance {
                            pos_yaw: [th_rel.x, th_rel.y, th_rel.z, sim_time * 2.4 + phase],
                            scale_pitch: [ring_s, 0.65, ring_s, 0.0],
                            tint_glow: [1.0, 0.88, 0.48, (1.0 - t_ring) * 0.82],
                            flags: [-1.0, 0.0, 0.0, 10.0],
                        });
                    }
                }
                CLASS_TIDEKIN => {
                    // Team footprint + shimmering aqua bioluminescent water-wake ring beneath!
                    let ring_rel = Vec3::new(wx, gy + 0.05, wz) - origin;
                    let rs = if is_self { 1.24 } else { 1.00 };
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y, ring_rel.z, 0.0],
                        scale_pitch: [rs, 1.0, rs, 0.0],
                        tint_glow: [ring_col[0], ring_col[1], ring_col[2], 0.70],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                    let wake_s = 1.24 + 0.24 * (sim_time * 3.6 + phase).sin();
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y + 0.03, ring_rel.z, -sim_time * 1.4 + phase],
                        scale_pitch: [wake_s, 0.72, wake_s, 0.0],
                        tint_glow: [0.24, 0.96, 1.0, 0.76],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                    if speed > 1.2 {
                        let w_cyc = (sim_time * 2.2 + phase * 0.5).fract();
                        let rip_s = 0.95 + w_cyc * 0.90;
                        let rip_rel = Vec3::new(
                            wx - p.vx * 0.055 * w_cyc,
                            gy + 0.04,
                            wz - p.vy * 0.055 * w_cyc,
                        ) - origin;
                        mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                            pos_yaw: [rip_rel.x, rip_rel.y, rip_rel.z, sim_time * 1.1],
                            scale_pitch: [rip_s, 0.52, rip_s, 0.0],
                            tint_glow: [0.36, 0.98, 1.0, (1.0 - w_cyc) * 0.55],
                            flags: [-1.0, 0.0, 0.0, 0.0],
                        });
                        vol_splats.push(VolSplat {
                            pos_rad: [rip_rel.x, rip_rel.y + 0.28, rip_rel.z, 0.36],
                            color_alpha: [0.32, 0.96, 1.08, 0.32 * (1.0 - w_cyc * 0.6)],
                            params: [0.85, 2.0, phase, 0.15],
                        });
                    }
                }
                CLASS_BURROWER => {
                    // Team footprint + glowing crystal dust & rock-spark wake ring beneath!
                    let ring_rel = Vec3::new(wx, gy + 0.05, wz) - origin;
                    let rs = if is_self { 1.28 } else { 1.06 };
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y, ring_rel.z, 0.0],
                        scale_pitch: [rs, 1.0, rs, 0.0],
                        tint_glow: [ring_col[0], ring_col[1], ring_col[2], 0.74],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                    // Rotating terracotta-amber / amethyst crystal dust & rock-spark wake ring
                    let drill_pulse = 1.30 + 0.22 * (sim_time * 4.8 + phase).sin();
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y + 0.04, ring_rel.z, sim_time * 2.6 + phase],
                        scale_pitch: [drill_pulse, 0.82, drill_pulse, 0.0],
                        tint_glow: [0.98, 0.58, 0.26, 0.80],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                    // Crystalline dirt dust + crisp amber & amethyst drill sparks around tunneling Burrowers!
                    let active_drill = speed > 0.9 || p.melee_active || p.dashing;
                    let spark_r = if active_drill { 1.05 } else { 0.74 };
                    let ang0 = sim_time * if active_drill { 7.4 } else { 3.1 } + phase;
                    let dust_rel = Vec3::new(
                        wx + ang0.cos() * spark_r - p.vx * 0.04,
                        gy + 0.30 + (sim_time * 7.6 + phase).sin().abs() * 0.32,
                        wz + ang0.sin() * spark_r - p.vy * 0.04,
                    ) - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [dust_rel.x, dust_rel.y, dust_rel.z, if active_drill { 0.48 } else { 0.32 }],
                        color_alpha: [1.28, 0.72, 0.24, if active_drill { 0.78 } else { 0.46 }],
                        params: [1.05, 4.0, phase + 1.3, 0.25],
                    });
                    let ang1 = ang0 + 2.094;
                    let crystal_rel = Vec3::new(
                        wx + ang1.cos() * (spark_r * 0.90),
                        gy + 0.44 + (sim_time * 9.2 + phase + 1.2).sin().abs() * 0.42,
                        wz + ang1.sin() * (spark_r * 0.90),
                    ) - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [crystal_rel.x, crystal_rel.y, crystal_rel.z, if active_drill { 0.38 } else { 0.26 }],
                        color_alpha: [0.98, 0.42, 1.35, if active_drill { 0.74 } else { 0.44 }],
                        params: [1.15, 4.0, phase + 3.7, 0.15],
                    });
                    if active_drill {
                        let ang2 = ang0 + 4.188;
                        let dirt_rel = Vec3::new(
                            wx + ang2.cos() * (spark_r * 1.12),
                            gy + 0.22 + (sim_time * 5.8 + phase).sin().abs() * 0.25,
                            wz + ang2.sin() * (spark_r * 1.12),
                        ) - origin;
                        vol_splats.push(VolSplat {
                            pos_rad: [dirt_rel.x, dirt_rel.y, dirt_rel.z, 0.46],
                            color_alpha: [0.68, 0.42, 0.26, 0.48],
                            params: [0.85, 2.0, phase + 5.1, 0.30],
                        });
                    }
                }
                _ => {
                    // CLASS_EMBERSTALKER (3): Compact Solo-Raider ember ground ring + ember aura!
                    let ring_rel = Vec3::new(wx, gy + 0.05, wz) - origin;
                    let solo_s = if is_self { 1.12 } else { 0.88 };
                    let solo_col = if is_self {
                        [1.0, 0.78, 0.24]
                    } else {
                        [1.0, 0.38, 0.14]
                    };
                    mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                        pos_yaw: [ring_rel.x, ring_rel.y, ring_rel.z, sim_time * 1.6 + phase],
                        scale_pitch: [solo_s, 1.08, solo_s, 0.0],
                        tint_glow: [solo_col[0], solo_col[1], solo_col[2], 0.84],
                        flags: [-1.0, 0.0, 0.0, 0.0],
                    });
                    // Compact orbiting solo-raider ember spark aura hugging the silhouette
                    let orb_a = sim_time * 3.8 + phase;
                    let orb_rel = Vec3::new(
                        wx + orb_a.cos() * 0.58,
                        gy + 1.15 + (sim_time * 5.2 + phase).sin() * 0.22,
                        wz + orb_a.sin() * 0.58,
                    ) - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [orb_rel.x, orb_rel.y, orb_rel.z, 0.34],
                        color_alpha: [1.32, 0.46, 0.12, 0.52],
                        params: [1.15, 0.0, phase, 0.12],
                    });
                }
            }

            if p.dashing || (p.status_flags & STATUS_SHIELDED) != 0 {
                mesh_batches[MeshKind::DashHalo as usize].push(MeshInstance {
                    pos_yaw: [rel.x, rel.y, rel.z, sim_time * 5.0],
                    scale_pitch: [1.25, 1.25, 1.25, 0.0],
                    tint_glow: [0.52, 0.94, 1.0, 0.92],
                    flags: [-1.0, 0.0, 0.0, 0.0],
                });
            }

            if p.melee_active {
                // MeleeArc mesh is centered along local +X, rotated to (cos(a), sin(a)) -> a = atan2(aim_y, aim_x)
                let arc_yaw = p.aim_y.atan2(p.aim_x);
                let slash_col = if p.class_id == CLASS_EMBERSTALKER {
                    [1.0, 0.44, 0.16]
                } else if p.class_id == CLASS_TIDEKIN {
                    [0.36, 0.98, 1.0]
                } else if p.class_id == CLASS_BURROWER {
                    [0.98, 0.60, 0.26]
                } else {
                    [1.0, 0.95, 0.65]
                };
                mesh_batches[MeshKind::MeleeArc as usize].push(MeshInstance {
                    pos_yaw: [rel.x, rel.y, rel.z, arc_yaw],
                    scale_pitch: [1.25, 1.0, 1.25, 0.0],
                    tint_glow: [slash_col[0], slash_col[1], slash_col[2], 0.95],
                    flags: [-1.0, 0.0, 0.0, 0.0],
                });
            }

            if p.ult_active {
                vol_splats.push(VolSplat {
                    pos_rad: [rel.x, rel.y + 1.5, rel.z, 3.2],
                    color_alpha: [team_tint[0] * 1.1, team_tint[1] * 1.1, team_tint[2] * 1.1, 0.65],
                    params: [1.2, 3.0, p.id as f32, 0.0],
                });
            }

            if p.carrying_relic {
                let bob_r = 0.22 * (sim_time * 3.8).sin();
                let r_rel = Vec3::new(wx, gy + 4.1 + bob_r, wz) - origin;
                mesh_batches[MeshKind::GoldenRelic as usize].push(MeshInstance {
                    pos_yaw: [r_rel.x, r_rel.y, r_rel.z, sim_time * 2.2],
                    scale_pitch: [0.98, 0.98, 0.98, 0.0],
                    tint_glow: [1.0, 0.94, 0.72, 0.45],
                    flags: [-1.0, 0.0, 0.0, 0.0],
                });
            }
        }

        // `G`-Key Squad Cohesion Tether Arcs: subtle translucent volumetric bead arcs
        // connecting nearby same-team non-solo allies (`dist < 14m`, capped at 24 arcs/frame)
        if show_squad_links && ent.players.len() >= 2 {
            const MAX_SQUAD_ARCS: usize = 24;
            const MAX_LINK_DIST: f32 = 14.0;
            const MIN_LINK_DIST: f32 = 0.6;
            const MAX_DEG_PER_ALLY: u8 = 2;

            let mut squad_nodes: Vec<(usize, u8, u32, Vec3)> = Vec::with_capacity(ent.players.len());
            for (idx, p) in ent.players.iter().enumerate() {
                if p.hp <= 0 || archetype_is_solo(p.class_id) {
                    continue;
                }
                let (wx, wz) = grid_to_xz(p.x, p.y);
                let gy = sample_e_y(p.x, p.y, p.z);
                squad_nodes.push((idx, p.team, p.id, Vec3::new(wx, gy + 1.45, wz)));
            }

            let link_cull_r2 = (cull_rad + 20.0) * (cull_rad + 20.0);
            let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
            for a in 0..squad_nodes.len() {
                let (_, team_a, _, pos_a) = squad_nodes[a];
                for b in (a + 1)..squad_nodes.len() {
                    let (_, team_b, _, pos_b) = squad_nodes[b];
                    if team_a != team_b {
                        continue;
                    }
                    let d = pos_a.distance(pos_b);
                    if d <= MIN_LINK_DIST || d >= MAX_LINK_DIST {
                        continue;
                    }
                    let mid = (pos_a + pos_b) * 0.5;
                    let cdx = mid.x - cam_target.x;
                    let cdz = mid.z - cam_target.z;
                    if cdx * cdx + cdz * cdz > link_cull_r2 {
                        continue;
                    }
                    candidates.push((d, a, b));
                }
            }
            candidates.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut deg = vec![0u8; squad_nodes.len()];
            for (dist, a, b) in candidates {
                if squad_arc_count >= MAX_SQUAD_ARCS {
                    break;
                }
                if deg[a] >= MAX_DEG_PER_ALLY || deg[b] >= MAX_DEG_PER_ALLY {
                    continue;
                }
                deg[a] += 1;
                deg[b] += 1;
                squad_arc_count += 1;

                let (_, team, id_a, pos_a) = squad_nodes[a];
                let (_, _, id_b, pos_b) = squad_nodes[b];
                let base_col = if team == 0 {
                    [0.34, 0.84, 1.18]
                } else {
                    [1.18, 0.44, 0.34]
                };
                let prox = (1.0 - (dist / MAX_LINK_DIST).powi(2)).clamp(0.28, 1.0);
                let beads = ((dist / 2.2).ceil() as usize).clamp(4, 7);
                let arch_h = (0.45 + dist * 0.085).min(1.65);
                let pair_seed = (id_a ^ id_b.rotate_left(7)) as f32 * 0.17;
                let flow = (sim_time * 0.55 + pair_seed).fract();

                for k in 0..beads {
                    let t = (k as f32 + 0.5) / beads as f32;
                    let lift = (t * std::f32::consts::PI).sin() * arch_h;
                    let wpos = pos_a.lerp(pos_b, t) + Vec3::new(0.0, lift, 0.0);
                    let rel = wpos - origin;
                    let d_flow = ((t - flow).abs()).min(1.0 - (t - flow).abs());
                    let pulse_boost = (1.0 - d_flow * 4.5).clamp(0.0, 1.0);
                    let alpha = (0.18 + 0.16 * pulse_boost) * prox;
                    let rad = 0.16 + 0.09 * pulse_boost;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, rad],
                        color_alpha: [
                            base_col[0] * (0.90 + 0.25 * pulse_boost),
                            base_col[1] * (0.90 + 0.25 * pulse_boost),
                            base_col[2] * (0.90 + 0.25 * pulse_boost),
                            alpha,
                        ],
                        params: [0.85, 3.0, pair_seed + k as f32 * 0.9, 0.0],
                    });
                }
            }
        }

        // Elemental Projectiles + Soft Volumetric Trails (in true 3D `(x, y, z)`!)
        for pr in &ent.projectiles {
            let (wx, wz) = grid_to_xz(pr.x, pr.y);
            let gy = sample_e_y(pr.x, pr.y, pr.z);
            let py = gy + 1.55;
            let rel = Vec3::new(wx, py, wz) - origin;
            let (col, lcol, vkind) = match pr.element {
                0 => ([1.15, 0.52, 0.14], [1.1, 0.52, 0.15], 0.0f32),
                1 => ([0.48, 0.94, 1.05], [0.38, 0.85, 1.15], 3.0f32),
                _ => ([0.68, 0.28, 0.88], [0.65, 0.28, 0.85], 1.0f32),
            };
            mesh_batches[MeshKind::ProjectileOrb as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, sim_time * 6.0 + pr.id as f32],
                scale_pitch: [pr.radius * 1.1, pr.radius * 1.1, pr.radius * 1.1, 0.0],
                tint_glow: [col[0], col[1], col[2], 1.0],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            vol_splats.push(VolSplat {
                pos_rad: [rel.x, rel.y, rel.z, pr.radius * 1.05],
                color_alpha: [col[0], col[1], col[2], 0.78],
                params: [1.0, vkind, pr.id as f32, 0.0],
            });
            push_light(Vec3::new(wx, py, wz), 9.0, lcol);
        }

        // Live 3D Volumetric Physical Actuator Beams, Cryo Fields, Matter Streams & Shockwaves!
        // In Z16 true-3D mode, endpoints interpolate directly from `(wx0, z0 + 1.1, wz0)` to
        // `(wx1, z1 + 1.0, wz1)` so aerial-to-bridge and ground-to-air diagonal beams shine in 3D!
        for (p_idx, pulse) in ent.actuator_pulses.iter().enumerate() {
            let (wx0, wz0) = grid_to_xz(pulse.x0, pulse.y0);
            let (wx1, wz1) = grid_to_xz(pulse.x1, pulse.y1);
            let (wy0, wy1) = if z16.is_some() {
                (pulse.z0 + 1.1, pulse.z1 + 1.0)
            } else {
                (
                    sample_e_y(pulse.x0, pulse.y0, pulse.z0) + 1.1,
                    sample_e_y(pulse.x1, pulse.y1, pulse.z1) + 1.0,
                )
            };
            let p0 = Vec3::new(wx0, wy0, wz0);
            let p1 = Vec3::new(wx1, wy1, wz1);
            let beam_vec = p1 - p0;
            let beam_len = beam_vec.length();
            let focus_r = pulse.focus_radius.clamp(1.5, 8.5);
            let seed = pulse.x0 * 0.73 + pulse.y0 * 1.19 + pulse.x1 * 0.41 + pulse.y1 * 0.89 + p_idx as f32 * 2.1;

            // True 3D orthonormal frame around the diagonal beam ray (safe even for steep vertical fire!)
            let beam_dir = beam_vec.normalize_or(Vec3::X);
            let up_ref = if beam_dir.y.abs() > 0.92 { Vec3::X } else { Vec3::Y };
            let perp_u = beam_dir.cross(up_ref).normalize_or(Vec3::X);
            let perp_v = perp_u.cross(beam_dir).normalize_or(Vec3::Y);

            // 1. Focused Heat Beam (thermal_flux > 0.2): vivid white-hot core + fiery orange-gold 3D diagonal plasma beam!
            if pulse.thermal_flux > 0.2 {
                let inten = pulse.thermal_flux.clamp(0.2, 1.0);
                let steps = ((beam_len / 0.95).ceil() as usize).clamp(6, 26);
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let wpos = p0.lerp(p1, t);
                    let rel = wpos - origin;
                    let rad = (0.34 + 0.22 * inten) * (0.72 + 0.42 * t);
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, rad],
                        color_alpha: [1.55, 0.62 + 0.26 * (1.0 - t), 0.12, (0.78 + 0.18 * inten).min(0.95)],
                        params: [1.05, 4.0, seed + s as f32 * 1.7, 0.25],
                    });
                }
                let rel_end = p1 - origin;
                // Outer scorching flare scaled by focus_radius
                vol_splats.push(VolSplat {
                    pos_rad: [rel_end.x, rel_end.y + 0.25, rel_end.z, focus_r * (0.48 + 0.18 * inten)],
                    color_alpha: [1.48, 0.50, 0.08, (0.70 + 0.18 * inten).min(0.90)],
                    params: [1.25, 0.0, seed + 21.0, 0.2],
                });
                // White-hot incandescent impact core sphere
                vol_splats.push(VolSplat {
                    pos_rad: [rel_end.x, rel_end.y + 0.20, rel_end.z, focus_r * (0.28 + 0.10 * inten)],
                    color_alpha: [1.50, 0.92, 0.36, 0.94],
                    params: [1.0, 4.0, seed + 31.0, 0.0],
                });
                push_light(
                    p1 + Vec3::new(0.0, 0.8, 0.0),
                    (focus_r * 2.8).clamp(8.0, 18.0),
                    [1.20 * inten, 0.58 * inten, 0.16 * inten],
                );
            }

            // 2. Cryo Field (thermal_flux < -0.2): razor-bright white-ice core + electric cyan 3D diagonal beam & crystals!
            if pulse.thermal_flux < -0.2 {
                let inten = (-pulse.thermal_flux).clamp(0.2, 1.0);
                let steps = ((beam_len / 1.15).ceil() as usize).clamp(6, 22);
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let ang = seed + s as f32 * 2.39996;
                    let cone_spread = focus_r * 0.28 * t;
                    let jitter = perp_u * (ang.cos() * cone_spread) + perp_v * (ang.sin() * cone_spread * 0.45);
                    let rel = (p0.lerp(p1, t) + jitter) - origin;
                    let is_crystal = s % 2 == 0;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, (0.34 + focus_r * 0.14 * t) * (0.78 + 0.26 * inten)],
                        color_alpha: if is_crystal {
                            [0.32, 0.94, 1.38, (0.74 + 0.20 * inten).min(0.94)]
                        } else {
                            [0.64, 0.96, 1.24, (0.56 + 0.20 * inten).min(0.80)]
                        },
                        params: [if is_crystal { 1.10 } else { 0.92 }, if is_crystal { 4.0 } else { 3.0 }, seed + s as f32 * 1.9, 0.15],
                    });
                }
                // Crystalline impact cloud ring scaled by focus_radius around true 3D impact point p1
                for k in 0..6 {
                    let ang = (k as f32 / 6.0) * std::f32::consts::TAU + seed * 0.4;
                    let rr = focus_r * 0.44;
                    let wpos = p1 + Vec3::new(ang.cos() * rr, 0.25 + (k % 2) as f32 * 0.35, ang.sin() * rr);
                    let rel = wpos - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, focus_r * (0.26 + 0.10 * inten)],
                        color_alpha: [0.38, 0.96, 1.32, (0.64 + 0.22 * inten).min(0.88)],
                        params: [1.05, if k % 2 == 0 { 4.0 } else { 2.0 }, seed + k as f32 * 2.3, 0.15],
                    });
                }
                push_light(
                    p1 + Vec3::new(0.0, 0.8, 0.0),
                    (focus_r * 2.6).clamp(8.0, 16.0),
                    [0.36 * inten, 0.86 * inten, 1.20 * inten],
                );
            }

            // 3A. Matter Stream — Oil (sp_oil > 0.2): luminous ultraviolet/magenta core + dark glossy oil arc
            if pulse.sp_oil > 0.2 {
                let inten = pulse.sp_oil.clamp(0.2, 1.0);
                let steps = ((beam_len / 1.35).ceil() as usize).clamp(5, 16);
                let horiz_factor = (1.0 - beam_dir.y.abs()).clamp(0.25, 1.0);
                for s in 1..=steps {
                    let t = s as f32 / steps as f32;
                    let arc_y = (t * std::f32::consts::PI).sin() * (1.1 + beam_len * 0.07) * horiz_factor;
                    let wobble = (seed + s as f32 * 3.1).sin() * (0.22 + focus_r * 0.09 * t);
                    let wpos = p0.lerp(p1, t) + Vec3::new(0.0, arc_y, 0.0) + perp_u * wobble;
                    let rel = wpos - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, 0.36 + 0.24 * inten],
                        color_alpha: [0.68, 0.22, 1.12, (0.76 + 0.16 * inten).min(0.92)],
                        params: [1.0, 4.0, seed + s as f32 * 2.7, 0.0],
                    });
                }
                for k in 0..4 {
                    let ang = (k as f32 / 4.0) * std::f32::consts::TAU + seed;
                    let rr = focus_r * 0.40;
                    let rel = (p1 + Vec3::new(ang.cos() * rr, 0.15, ang.sin() * rr)) - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, 0.42 + focus_r * 0.12],
                        color_alpha: [0.42, 0.14, 0.72, (0.72 + 0.18 * inten).min(0.88)],
                        params: [0.85, 3.0, seed + k as f32 * 4.1, 0.0],
                    });
                }
            }

            // 3B. Matter Stream — Water (sp_water > 0.2): azure mist stream + spray cloud along true 3D ray
            if pulse.sp_water > 0.2 {
                let inten = pulse.sp_water.clamp(0.2, 1.0);
                let steps = ((beam_len / 1.65).ceil() as usize).clamp(4, 14);
                for s in 1..=steps {
                    let t = s as f32 / steps as f32;
                    let wobble = (seed + s as f32 * 2.1).cos() * (0.30 + focus_r * 0.14 * t);
                    let rel = (p0.lerp(p1, t) + perp_u * wobble) - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, 0.42 + focus_r * 0.15 * t],
                        color_alpha: [0.26, 0.76, 1.08, (0.50 + 0.22 * inten).min(0.76)],
                        params: [1.0, 2.0, seed + s as f32 * 1.6, 0.3],
                    });
                }
                let rel_end = p1 - origin;
                vol_splats.push(VolSplat {
                    pos_rad: [rel_end.x, rel_end.y + 0.4, rel_end.z, focus_r * (0.42 + 0.15 * inten)],
                    color_alpha: [0.34, 0.82, 1.12, (0.52 + 0.20 * inten).min(0.74)],
                    params: [1.10, 2.0, seed + 13.0, 0.3],
                });
            }

            // 3C. Matter Stream — Smoke (sp_smoke > 0.2): billowing gray-indigo smokescreen puffs
            if pulse.sp_smoke > 0.2 {
                let inten = pulse.sp_smoke.clamp(0.2, 1.0);
                for k in 0..7 {
                    let ang = (k as f32 / 7.0) * std::f32::consts::TAU + seed * 0.7;
                    let rr = if k == 0 { 0.0 } else { focus_r * (0.28 + 0.28 * ((k % 3) as f32 * 0.5)) };
                    let wpos = if k >= 5 {
                        p0.lerp(p1, 0.55 + (k - 5) as f32 * 0.20)
                    } else {
                        p1 + Vec3::new(ang.cos() * rr, 0.45 + (k % 3) as f32 * 0.45, ang.sin() * rr)
                    };
                    let rel = wpos - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, focus_r * (0.30 + 0.16 * inten)],
                        color_alpha: [0.30, 0.31, 0.44, (0.42 + 0.20 * inten).min(0.66)],
                        params: [1.10, 1.0, seed + k as f32 * 3.3, 0.45],
                    });
                }
            }

            // 4. Kinetic Impulse (impulse > 0.2): translucent expanding shockwave ring splats around (x1, y1, z1)
            if pulse.impulse > 0.2 {
                let inten = pulse.impulse.clamp(0.2, 1.0);
                let wave_phase = (sim_time * 3.6 + seed * 0.25).fract();
                let ring_r = (focus_r * 0.45 + wave_phase * (focus_r * 0.85 + 2.4)).max(1.2);
                let ring_col = if pulse.team == 0 {
                    [0.64, 0.92, 1.18]
                } else {
                    [1.18, 0.80, 0.52]
                };
                let ring_alpha = ((1.0 - wave_phase * 0.65) * (0.48 + 0.30 * inten)).clamp(0.18, 0.78);
                let ring_n = 12usize;
                for k in 0..ring_n {
                    let ang = (k as f32 / ring_n as f32) * std::f32::consts::TAU + wave_phase * 0.6;
                    let wpos = p1 + Vec3::new(ang.cos() * ring_r, 0.25, ang.sin() * ring_r);
                    let rel = wpos - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, 0.42 + 0.34 * inten],
                        color_alpha: [ring_col[0], ring_col[1], ring_col[2], ring_alpha],
                        params: [0.68, 3.0, seed + k as f32 * 0.8, 0.0],
                    });
                }
            }
        }

        // Live 3D Volumetric Spatial Time-Dilation Domes (`WorldModelDT`: 0.15x Ice-Cyan Stasis / 3.0x Amber-Gold Haste!)
        for dome in &ent.time_domes {
            let (wx, wz) = grid_to_xz(dome.x, dome.y);
            let gy = sample_g_y(dome.x, dome.y);
            let rel_base = Vec3::new(wx, gy + 0.14, wz) - origin;
            let is_stasis = dome.dt_scale < 0.8;
            let fade = (dome.ttl_s / 2.0).clamp(0.18, 1.0);
            let pulse = 0.92 + 0.08 * (sim_time * if is_stasis { 1.4 } else { 5.8 } + dome.id as f32).sin();
            let spin = sim_time * if is_stasis { 0.22 } else { 2.10 } + dome.id as f32 * 0.9;
            let (ring_col, wisp_col, light_col) = if is_stasis {
                ([0.24, 0.92, 1.20], [0.42, 0.96, 1.28], [0.22, 0.80, 1.12])
            } else {
                ([1.22, 0.78, 0.18], [1.36, 0.88, 0.28], [1.12, 0.70, 0.18])
            };

            let r = dome.radius.clamp(6.0, 28.0);
            // 1. Sleek counter-rotating runic latitude rings framing the Time-Dilation Dome hemisphere!
            let ring_tiers = [
                (0.0f32, 1.00f32, 0.92f32 * fade, 24.0f32),
                (r * 0.25, 0.88, 0.78 * fade, 18.0f32),
                (r * 0.48, 0.62, 0.66 * fade, 14.0f32),
            ];
            for (ti, &(dy, r_mul, glow_a, segs)) in ring_tiers.iter().enumerate() {
                let s = r * r_mul;
                let dir = if ti % 2 == 0 { 1.0 } else { -1.0 };
                mesh_batches[MeshKind::ShrineRuneRing as usize].push(MeshInstance {
                    pos_yaw: [rel_base.x, rel_base.y + dy, rel_base.z, spin * dir],
                    scale_pitch: [s, 1.15, s, 0.0],
                    tint_glow: [ring_col[0] * pulse, ring_col[1] * pulse, ring_col[2] * pulse, glow_a],
                    flags: [-1.0, 0.0, 0.0, segs],
                });
            }

            // 2. Pulsing Hexagonal Energy-Grid Volumetric Shell (`kind = 5.0`) forming a structured 3D geodesic dome!
            let hemi_h = r * 0.66;
            let bands: [(usize, f32, f32); 3] = [
                (14, 0.16, r * 0.22),
                (10, 0.44, r * 0.24),
                (6, 0.72, r * 0.25),
            ];
            for (bi, &(cnt, el_frac, rad_w)) in bands.iter().enumerate() {
                let el = el_frac * std::f32::consts::FRAC_PI_2;
                let ring_r = r * el.cos();
                let band_y = gy + 0.55 + hemi_h * el.sin();
                for k in 0..cnt {
                    let az = (k as f32 / cnt as f32) * std::f32::consts::TAU
                        + spin * (0.35 + bi as f32 * 0.22);
                    let wpos = Vec3::new(wx + az.cos() * ring_r, band_y, wz + az.sin() * ring_r);
                    let rel = wpos - origin;
                    vol_splats.push(VolSplat {
                        pos_rad: [rel.x, rel.y, rel.z, rad_w.clamp(1.2, 3.8)],
                        color_alpha: [
                            wisp_col[0] * pulse,
                            wisp_col[1] * pulse,
                            wisp_col[2] * pulse,
                            (if is_stasis { 0.54 } else { 0.62 }) * fade,
                        ],
                        params: [1.05, 5.0, dome.id as f32 * 2.3 + k as f32 * 1.1 + bi as f32 * 0.7, 0.0],
                    });
                }
            }
            // Zenith hexagonal crown energy seal
            let crown_rel = Vec3::new(wx, gy + 0.75 + hemi_h, wz) - origin;
            vol_splats.push(VolSplat {
                pos_rad: [crown_rel.x, crown_rel.y, crown_rel.z, (r * 0.28).clamp(1.8, 4.4)],
                color_alpha: [wisp_col[0] * pulse, wisp_col[1] * pulse, wisp_col[2] * pulse, 0.64 * fade],
                params: [0.85, 5.0, dome.id as f32 * 4.1, 0.0],
            });
            push_light(
                Vec3::new(wx, gy + 4.8, wz),
                (r * 1.6).clamp(14.0, 28.0),
                [light_col[0] * fade * pulse, light_col[1] * fade * pulse, light_col[2] * fade * pulse],
            );
        }
    }

    // Live 3D Ground Aim Reticle under the player's cursor (shows exact aim point & focus radius!)
    if let Some(ret) = aim_reticle {
        let (rx, rz) = grid_to_xz(ret.gx, ret.gy);
        let ry = sample_e_y(ret.gx, ret.gy, ret.gz) + 0.08;
        let rel = Vec3::new(rx, ry, rz) - origin;
        let outer_s = (ret.radius * 0.72).clamp(1.2, 6.0);
        // Outer focus-radius ring
        mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
            pos_yaw: [rel.x, rel.y, rel.z, sim_time * 0.7],
            scale_pitch: [outer_s, 0.85, outer_s, 0.0],
            tint_glow: [ret.color[0], ret.color[1], ret.color[2], 0.78],
            flags: [-1.0, 0.0, 0.0, 0.0],
        });
        // Crisp inner aim core ring
        mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
            pos_yaw: [rel.x, rel.y + 0.04, rel.z, -sim_time * 1.4],
            scale_pitch: [0.55, 1.15, 0.55, 0.0],
            tint_glow: [1.0, 0.98, 0.90, 0.95],
            flags: [-1.0, 0.0, 0.0, 0.0],
        });
        vol_splats.push(VolSplat {
            pos_rad: [rel.x, rel.y + 0.45, rel.z, 0.62],
            color_alpha: [ret.color[0], ret.color[1], ret.color[2], 0.45],
            params: [1.0, 3.0, sim_time * 2.0, 0.0],
        });
    }

    // Subtle atmospheric drifting snow / frost / ember motes around the camera for living depth
    let weather_kind = entities.map(|e| e.telemetry.weather_kind).unwrap_or(0);
    let mote_col = match weather_kind {
        3 => [1.18, 0.62, 0.24], // Ember dusk sparks
        1 => [0.88, 0.95, 1.05], // Blizzard snow flurries
        _ => [0.78, 0.90, 1.02], // Crisp frost air motes
    };
    let mote_cnt = if weather_kind == 1 { 36 } else { 20 };
    for m in 0..mote_cnt {
        let h0 = hash2i(m * 37 + 11, 913);
        let h1 = hash2i(m * 73 + 29, 419);
        let h2 = hash2i(m * 19 + 5, 733);
        let drift_x = (h0 * 76.0 + sim_time * (2.6 + h2 * 2.4)).rem_euclid(76.0) - 38.0;
        let drift_z = (h1 * 76.0 + sim_time * (1.2 + h0 * 1.5)).rem_euclid(76.0) - 38.0;
        let mote_span_y = if z_depth >= 48 { 28.0 } else { 14.0 };
        let base_y = (cam_target.y - mote_span_y * 0.4).max(1.2);
        let fall_y = (mote_span_y - (h2 * mote_span_y + sim_time * (1.4 + h1 * 1.2)).rem_euclid(mote_span_y)) + base_y;
        let rel = Vec3::new(cam_target.x + drift_x, fall_y, cam_target.z + drift_z) - origin;
        vol_splats.push(VolSplat {
            pos_rad: [rel.x, rel.y, rel.z, if weather_kind == 1 { 0.28 } else { 0.20 }],
            color_alpha: [mote_col[0], mote_col[1], mote_col[2], if weather_kind == 1 { 0.48 } else { 0.34 }],
            params: [1.0, 3.0, m as f32 * 1.9, 0.4],
        });
    }

    // Back-to-front depth sort ONLY the compact soft volumetric splats (~300–1200) for clean alpha compositing!
    vol_splats.sort_by(|a, b| {
        let da = (Vec3::from_slice(&a.pos_rad[..3]) - mats.rel_eye).length_squared();
        let db = (Vec3::from_slice(&b.pos_rad[..3]) - mats.rel_eye).length_squared();
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    AssembledFrame {
        terrain_verts,
        terrain_idxs,
        mesh_batches,
        surfel_placements,
        surfel_refs,
        vol_splats,
        point_pos,
        point_col,
        visible_chunks,
        squad_arc_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::MobaCamera;
    use crate::hud::{build_hud, HudConfig};
    use crate::mesh_gen::{archetype_color, archetype_tag, StaticMeshBank};
    use frostmarsh_proto::{ActuatorPulseSnap, BotBrainSnap, PlayerSnap, TelemetrySnap};

    fn make_mock_entities(player_count: usize) -> LiveEntityState {
        let mut players = Vec::with_capacity(player_count);
        for i in 0..player_count {
            let class_id = (i % 5) as u8;
            let team = (i % 2) as u8;
            // Cluster allies in compact squads so squad cohesion arcs (<14m) engage naturally
            let squad_idx = i / 6;
            let slot_in_squad = i % 6;
            let base_angle = (squad_idx as f32 / ((player_count + 5) / 6).max(1) as f32) * std::f32::consts::TAU;
            let squad_rad = 10.0 + (squad_idx % 3) as f32 * 9.0;
            let sub_angle = (slot_in_squad as f32 / 6.0) * std::f32::consts::TAU;
            let x = (128.0 + base_angle.cos() * squad_rad + sub_angle.cos() * 3.8).clamp(6.0, 250.0);
            let y = (128.0 + base_angle.sin() * squad_rad + sub_angle.sin() * 3.8).clamp(6.0, 250.0);
            let z = match class_id {
                CLASS_SKYWING => 2.2 + (i % 3) as f32 * 0.4,
                CLASS_EMBERSTALKER => 1.6,
                CLASS_BURROWER => 0.65,
                _ => 1.0,
            };
            players.push(PlayerSnap {
                id: (i + 1) as u32,
                name: format!("Hero{:02}", i + 1),
                team,
                class_id,
                is_bot: i != 0,
                x,
                y,
                z,
                vx: base_angle.sin() * 14.0,
                vy: -base_angle.cos() * 14.0,
                vz: if class_id == CLASS_SKYWING { 0.8 } else { 0.0 },
                aim_x: base_angle.cos(),
                aim_y: base_angle.sin(),
                hp: 110,
                max_hp: 140,
                mana: 85,
                max_mana: 120,
                stamina: 90,
                max_stamina: 100,
                level: (1 + (i % 5)) as u8,
                xp: 50,
                kills: (i % 7) as u16,
                deaths: (i % 4) as u16,
                status_flags: 0,
                dashing: i % 11 == 0,
                melee_active: i % 9 == 0,
                ult_active: i % 16 == 0,
                carrying_relic: i == 5,
                dash_cd_frac: 0.0,
                ability_e_cd_frac: 0.25,
                ult_r_cd_frac: 0.5,
                ack_input_seq: 1,
                score: (player_count - i) as u32 * 10,
                face: 0,
            });
        }
        let bot_brains = vec![BotBrainSnap {
            bot_id: 5,
            team: 0,
            class_id: CLASS_BURROWER,
            z_heatmap_4x4: [120; 16],
            elem_probs: [200, 90, 40],
            ability_probs: [140, 180, 120, 60, 210, 95],
            target_id: 2,
        }];
        LiveEntityState {
            sim_tick: 120,
            match_time_secs: 95,
            your_player_id: 1,
            echo_client_time_ms: 10,
            relic_x: 128.0,
            relic_y: 128.0,
            players,
            projectiles: Vec::new(),
            bodies: Vec::new(),
            shrines: Vec::new(),
            events: Vec::new(),
            telemetry: TelemetrySnap {
                mode: "neural".to_string(),
                backend: "cuda".to_string(),
                ecosystem_on: true,
                grid_w: 256,
                grid_h: 256,
                grid_d: 4,
                field_tick: 120,
                infer_ms_last: 1.1,
                infer_ms_p99: 1.4,
                eco_infer_ms: 0.3,
                proj_active_pct: 42.0,
                mass_error: 0,
                energy_error: 0,
                wire_kbps: 210.0,
                aoi_chunks_sent: 16,
                total_chunks: 256,
                active_bots: player_count.saturating_sub(1) as u16,
                neural_bots_on: true,
                bot_infer_ms: 0.8,
                weather_kind: 0,
                day_time: 0.45,
                wind_x: 0.5,
                wind_y: 0.2,
                score_west: 180,
                score_east: 165,
                acoustic_ms: 0.4,
                world_sparse_pct: 64.0,
                physics_native: false,
                time_scale: 1.0,
                world_day: 1,
                neural_3d_vertical: true,
                active_time_domes: 0,
                system2_slm_active: false,
                planet_mode: false,
                planet_r_core: 0.0,
                sun_dir: [0.0, 1.0, 0.0],
                seam_mass_crossed: 0,
                active_sky_cores: 0,
                macro_planet: false,
                planet_shards_per_axis: 0,
                planet_seed: 0,
                climate_infer_ms: 0.0,
                active_hydrated_shards: 0,
                local_wind_u: 0.0,
                local_wind_v: 0.0,
                local_temp_c: 0.0,
                local_cloud_cover: 0.0,
            },
            bot_brains,
            kill_feed: Vec::new(),
            speech_bubbles: Vec::new(),
            actuator_pulses: Vec::new(),
            match_phase: 0,
            match_time_left_s: 205,
            mvp_name: "Hero01".to_string(),
            mvp_score: 480,
            time_domes: Vec::new(),
        }
    }

    #[test]
    fn test_five_archetype_meshes_64p_diagonal_beams_and_squad_links() {
        assert_eq!(archetype_tag(CLASS_BURROWER), "BURROWER");
        assert!(!archetype_is_solo(CLASS_BURROWER));
        assert!(archetype_is_solo(CLASS_EMBERSTALKER));
        assert!(archetype_color(CLASS_BURROWER).iter().all(|c| c.is_finite()));

        let bank = StaticMeshBank::build(1);
        let archetype_kinds = [
            MeshKind::HeroStrider,
            MeshKind::HeroSkywing,
            MeshKind::SkywingWingL,
            MeshKind::SkywingWingR,
            MeshKind::HeroTidekin,
            MeshKind::HeroStalker,
            MeshKind::HeroBurrower,
        ];
        for kind in archetype_kinds {
            let sl = bank.slices[kind as usize];
            assert!(sl.index_count > 0, "{:?} must produce non-empty geometry", kind);
            let start = sl.first_index as usize;
            let end = start + sl.index_count as usize;
            assert!(end <= bank.indices.len());
            for &idx in &bank.indices[start..end] {
                let v = &bank.vertices[idx as usize];
                assert!(v.pos.iter().all(|c| c.is_finite()), "{:?} has non-finite pos", kind);
                assert!(v.normal.iter().all(|c| c.is_finite()), "{:?} has non-finite normal", kind);
                assert!(v.color_mat.iter().all(|c| c.is_finite()), "{:?} has non-finite color", kind);
            }
        }

        // Verify 48-player AND 64-player assembly across all 5 archetypes + Squad Links + HUD scaling
        let world = LiveApproach4World::new(256, 256, 4);
        let surfels = SurfelProtoBank::build();
        let cam = MobaCamera {
            target: Vec3::new(0.0, 2.0, 0.0),
            yaw: 0.42,
            pitch: 0.58,
            dist: 52.0,
            fov_y: 38.0_f32.to_radians(),
            ortho: false,
        };
        let mats = cam.compute_matrices(1920.0, 1080.0, Vec3::new(0.4, 0.8, 0.4).normalize());

        for player_count in [48usize, 64usize] {
            let ent = make_mock_entities(player_count);
            let frame = assemble_live_frame(
                &world,
                None,
                Some(&ent),
                &surfels,
                &mats,
                cam.target,
                cam.dist,
                3.25,
                0,
                None,
                true,
            );
            let frame_no_links = assemble_live_frame(
                &world,
                None,
                Some(&ent),
                &surfels,
                &mats,
                cam.target,
                cam.dist,
                3.25,
                0,
                None,
                false,
            );

            let expected = |cls: usize| (player_count + 4 - cls) / 5;
            assert_eq!(frame.mesh_batches[MeshKind::HeroStrider as usize].len(), expected(0));
            assert_eq!(frame.mesh_batches[MeshKind::HeroSkywing as usize].len(), expected(1));
            assert_eq!(frame.mesh_batches[MeshKind::SkywingWingL as usize].len(), expected(1));
            assert_eq!(frame.mesh_batches[MeshKind::SkywingWingR as usize].len(), expected(1));
            assert_eq!(frame.mesh_batches[MeshKind::HeroTidekin as usize].len(), expected(2));
            assert_eq!(frame.mesh_batches[MeshKind::HeroStalker as usize].len(), expected(3));
            assert_eq!(frame.mesh_batches[MeshKind::HeroBurrower as usize].len(), expected(4));
            assert!(frame.mesh_batches[MeshKind::GroundRing as usize].len() >= player_count);
            assert!(!frame.vol_splats.is_empty());

            // Squad cohesion tether arcs: active when enabled, capped <= 24, disabled when off
            assert!(frame.squad_arc_count > 0, "expected active squad links for {}p", player_count);
            assert!(frame.squad_arc_count <= 24, "squad links must be capped at 24");
            assert_eq!(frame_no_links.squad_arc_count, 0);
            assert!(frame.vol_splats.len() > frame_no_links.vol_splats.len());

            for batch in &frame.mesh_batches {
                for inst in batch {
                    assert!(inst.pos_yaw.iter().all(|v| v.is_finite()));
                    assert!(inst.scale_pitch.iter().all(|v| v.is_finite()));
                    assert!(inst.tint_glow.iter().all(|v| v.is_finite()));
                    assert!(inst.flags.iter().all(|v| v.is_finite()));
                }
            }
            for splat in &frame.vol_splats {
                assert!(splat.pos_rad.iter().all(|v| v.is_finite()));
                assert!(splat.color_alpha.iter().all(|v| v.is_finite()));
                assert!(splat.params.iter().all(|v| v.is_finite()));
            }

            let hud_batch = build_hud(
                1920.0,
                1080.0,
                &world,
                None,
                Some(&ent),
                &mats,
                &HudConfig {
                    show_hud: true,
                    show_telemetry: true,
                    show_brain_inspector: true,
                    brain_selected_idx: Some(0),
                    selected_element: 0,
                    focus_radius: 4.0,
                    style_label: "PIXEL-2XSS",
                    z16_active: false,
                    z16_style_label: "NORDIC",
                    fps: 120.0,
                    rtt_ms: 2.0,
                    chat_open: false,
                    chat_input: "",
                    chat_shout: true,
                    show_scoreboard: true,
                    audio_muted: false,
                    audio_available: true,
                    cave_enclosure: 0.0,
                    show_squad_links: true,
                    camera_preset_label: "TACTICAL",
                    macro_eye_f64: None,
                },
            );
            assert!(!hud_batch.vertices.is_empty());
            for hv in &hud_batch.vertices {
                assert!(hv.pos[0].is_finite() && hv.pos[1].is_finite());
                assert!(hv.color.iter().all(|c| c.is_finite()));
            }
        }

        // Verify Solo Emberstalkers are NEVER linked by squad cohesion tethers
        let mut solo_ent = make_mock_entities(6);
        for (i, p) in solo_ent.players.iter_mut().enumerate() {
            p.class_id = CLASS_EMBERSTALKER;
            p.team = 0;
            p.x = 128.0 + (i as f32) * 1.5;
            p.y = 128.0;
        }
        let solo_frame = assemble_live_frame(
            &world,
            None,
            Some(&solo_ent),
            &surfels,
            &mats,
            cam.target,
            cam.dist,
            1.0,
            0,
            None,
            true,
        );
        assert_eq!(solo_frame.squad_arc_count, 0, "solo Emberstalkers must never be tethered");

        // Verify True 3D Diagonal Vertical Actuator Beams (z0 -> z1) in Z16 mode:
        // Skywing firing heat beam diagonally DOWN (z0=11.0 -> z1=4.0) &
        // Strider firing cryo beam diagonally UP (z0=2.0 -> z1=11.5)
        let z16 = Z16World::new(128, 128, 16);
        let mut beam_ent = make_mock_entities(5);
        beam_ent.players.clear();
        beam_ent.actuator_pulses.push(ActuatorPulseSnap {
            x0: 64.0,
            y0: 64.0,
            z0: 11.0,
            x1: 72.0,
            y1: 68.0,
            z1: 4.0,
            thermal_flux: 0.95,
            sp_oil: 0.0,
            sp_water: 0.0,
            sp_smoke: 0.0,
            focus_radius: 3.5,
            impulse: 0.0,
            team: 0,
            face: 0,
        });
        beam_ent.actuator_pulses.push(ActuatorPulseSnap {
            x0: 60.0,
            y0: 60.0,
            z0: 2.0,
            x1: 66.0,
            y1: 64.0,
            z1: 11.5,
            thermal_flux: -0.90,
            sp_oil: 0.0,
            sp_water: 0.0,
            sp_smoke: 0.0,
            focus_radius: 3.5,
            impulse: 0.0,
            team: 1,
            face: 0,
        });
        let beam_frame = assemble_live_frame(
            &world,
            Some(&z16),
            Some(&beam_ent),
            &surfels,
            &mats,
            cam.target,
            cam.dist,
            2.0,
            0,
            None,
            false,
        );
        // Convert relative splat Y back to world Y using mats.origin.y
        let beam_ys: Vec<f32> = beam_frame
            .vol_splats
            .iter()
            .filter(|s| s.pos_rad[3] >= 0.31) // exclude small ambient weather motes (0.20)
            .map(|s| s.pos_rad[1] + mats.origin.y)
            .collect();
        assert!(
            beam_ys.iter().any(|&y| y >= 11.0),
            "diagonal 3D beams must reach airborne z0/z1 altitude >= 11.0m, got max {:?}",
            beam_ys.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        );
        assert!(
            beam_ys.iter().any(|&y| y <= 4.5),
            "diagonal 3D beams must reach ground/bridge z0/z1 altitude <= 4.5m, got min {:?}",
            beam_ys.iter().copied().fold(f32::INFINITY, f32::min)
        );
        assert!(
            beam_ys.iter().any(|&y| y > 6.0 && y < 9.5),
            "diagonal 3D beams must interpolate smoothly through mid-air altitude"
        );
    }
}
