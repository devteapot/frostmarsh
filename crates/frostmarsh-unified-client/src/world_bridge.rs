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
use crate::mesh_gen::{MeshInstance, MeshKind, MeshVertex, MESH_KIND_COUNT};
use crate::net::LiveEntityState;
use crate::parametric::hash2i;
use crate::surfel_gen::{SplatPlacement, SplatRef, SurfelProtoBank, SurfelProtoKind};
use bytemuck::{Pod, Zeroable};
use frostmarsh_proto::{
    chunks_for_dim, CHUNK_SIZE, CLASS_ALCHEMIST, CLASS_WARDEN, STATUS_SHIELDED,
};
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
    let gentle_hill = 0.26 * (nx * 11.0).sin() * (ny * 9.0).cos() + (1.0 - ny) * 0.42;

    let h = if c.s1 == SP_ROCK || (c.wall && c.s1 != SP_WOOD) {
        if gx < 4 || gx + 4 >= grid_w || gy < 4 || gy + 4 >= grid_h {
            3.1 + gentle_hill * 0.4
        } else {
            2.35 + gentle_hill * 0.35
        }
    } else if is_bridge {
        0.58
    } else if is_ice {
        0.30
    } else if is_water {
        -0.48
    } else if c.s1 == SP_OIL && c.s2 == SP_WATER {
        -0.36
    } else if c.s1 == SP_OIL || (c.s2 == SP_OIL && c.s2_frac > 0.25) {
        0.58 + gentle_hill * 0.2
    } else if c.s1 == SP_ASH {
        0.74 + gentle_hill * 0.4
    } else {
        0.94 + gentle_hill
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
        let sheen = 0.5 + 0.5 * ((gx as f32 * 0.6 + gy as f32 * 0.4).sin());
        rgb = [
            rgb[0] * 0.75 + 0.14 * sheen,
            rgb[1] * 0.75 + 0.08 * (1.0 - sheen),
            rgb[2] * 0.75 + 0.24 * sheen,
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
            self.compute_separable_top_weighted_normals(rx0, ry0, rx1, ry1);

            for &ci in &surf_dirty {
                self.rebuild_chunk(ci % cx_dim, ci / cx_dim);
            }
        }

        // --- Layer 2 (Highland Basalt Plateaus, Overhanging Skybridges, Alpine Lakes & 3D Waterfalls!) ---
        if !l2_dirty.is_empty() && 2 < aplus_layers.len() {
            let raw2 = &aplus_layers[2];
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
                self.rebuild_layer2_chunk(cx, cy);
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
        };

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

                // Mid-air rising Steam / Smoke on Layer 2 -> Soft Volumetric 3DGS!
                if c.s1 == SP_STEAM || c.s1 == SP_SMOKE {
                    if j0 < 0.35 {
                        cache.volumetrics.push(ChunkVolItem {
                            world_pos: Vec3::new(wx, 4.6 + j1 * 1.1, wz),
                            radius: 1.25 + j1 * 0.65,
                            height_stretch: 1.15,
                            color: if c.s1 == SP_STEAM { [0.88, 0.94, 0.99] } else { [0.22, 0.20, 0.28] },
                            alpha: if c.s1 == SP_STEAM { 0.52 } else { 0.38 },
                            kind: if c.s1 == SP_STEAM { 2.0 } else { 1.0 },
                            phase: j0 * 14.0,
                        });
                    }
                    continue;
                }

                // Determine Layer 2 vertical slab height & overhang thickness!
                let is_skybridge_wood = c.s1 == SP_WOOD && !c.wall && c.walkable;
                let is_water_l2 = c.s1 == SP_WATER;
                let is_overhang_arch = self.decoded[idx].s1 == SP_WATER && !is_water_l2;

                let top_y = if is_water_l2 { 4.05 } else if is_skybridge_wood { 4.35 } else { 4.45 };
                // Overhanging skybridges & stone arches are 0.75m thick so you see right underneath them!
                // Solid basalt plateaus extend all the way down into the ground (`bot_y = 0.4`)!
                let bot_y = if is_overhang_arch || is_skybridge_wood { 3.55 } else { 0.40 };

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
                            // Animated plunging 3D Waterfall Curtain all the way down to fjord!
                            let mut wp0 = p0; let mut wp3 = p3;
                            wp0[1] = 0.05; wp3[1] = 0.05;
                            Self::push_box_quad(
                                &mut cache.terrain_verts,
                                &mut cache.terrain_idxs,
                                wp0, p1, p2, wp3,
                                norm,
                                [0.56, 0.88, 1.0, 3.0],
                                [c.temp_norm, 0.0, 0.0, 2.2],
                            );
                            waterfall_edge = true;
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

                // Billowing 3D waterfall splash mist at the base of the cascade!
                if waterfall_edge && j0 < 0.65 {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, 1.15 + j1 * 0.9, wz),
                        radius: 1.55 + j1 * 0.65,
                        height_stretch: 1.25,
                        color: [0.82, 0.94, 1.0],
                        alpha: 0.56,
                        kind: 2.0,
                        phase: j0 * 16.0,
                    });
                }

                // Burning Layer 2 Timber Skybridge / Oil Cascade -> Soft Volumetric Fire!
                if c.burning && j0 < 0.55 {
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

                // Highland Snow-Pine Surfels perched up on Layer 2 plateaus!
                if !is_skybridge_wood && c.s1 == SP_WOOD && c.wall {
                    let sc = 0.72 + j0 * 0.35;
                    cache.props.push(ChunkPropItem {
                        kind: MeshKind::TreeTrunk,
                        world_pos: Vec3::new(wx, top_y - 0.05, wz),
                        yaw: j0 * 6.28,
                        scale: Vec3::splat(sc),
                        tint: [1.0, 1.0, 1.0],
                        glow: 0.0,
                    });
                    cache.surfels.push(ChunkSurfelItem {
                        kind: SurfelProtoKind::SnowPine,
                        world_pos: Vec3::new(wx, top_y, wz),
                        yaw: j1 * 6.28,
                        scale: sc,
                        tint: [0.96, 0.99, 1.04],
                        wind_phase: gx as f32 * 0.4 + gy as f32 * 0.5,
                        glow_mul: 0.0,
                    });
                }
            }
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
                if j0 > 0.34 {
                    continue;
                }
                let (wx, wz) = self.grid_to_world_xz(gx as f32 + 0.5, gy as f32 + 0.5);
                if c.s1 == SP_STEAM || c.s2 == SP_STEAM {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, 7.6 + j1 * 2.2, wz),
                        radius: 2.1 + j1 * 1.1,
                        height_stretch: 0.85,
                        color: [0.88, 0.94, 1.0],
                        alpha: 0.42,
                        kind: 2.0,
                        phase: j0 * 20.0,
                    });
                } else if c.s1 == SP_SMOKE || c.s2 == SP_SMOKE {
                    cache.volumetrics.push(ChunkVolItem {
                        world_pos: Vec3::new(wx, 8.2 + j1 * 2.4, wz),
                        radius: 2.2 + j0 * 1.2,
                        height_stretch: 0.90,
                        color: [0.20, 0.18, 0.28],
                        alpha: 0.36,
                        kind: 1.0,
                        phase: j1 * 18.0,
                    });
                }
            }
        }
        self.chunks_l3[ci] = cache;
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
        };

        // 1. Smoothed terrain grid
        for ly in 0..vy_n {
            let gy = y0 + ly;
            for lx in 0..vx_n {
                let gx = x0 + lx;
                let idx = gy * gw + gx;
                let c = &self.decoded[idx];
                let (h, rgb, _is_water, burn_t) = cell_height_and_color(c, gx, gy, gw, gh);
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
                let (h, _, is_water, burn_t) = cell_height_and_color(c, gx, gy, gw, gh);
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
                    if c.s1 == SP_WOOD || c.s2 == SP_WOOD {
                        cache.props.push(ChunkPropItem {
                            kind: MeshKind::TreeTrunk,
                            world_pos: Vec3::new(wx, h - 0.05, wz),
                            yaw: j0 * 6.28,
                            scale: Vec3::splat(0.76 + j1 * 0.24),
                            tint: [0.32, 0.20, 0.17],
                            glow: 0.12,
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

                // APPROACH 4 HARD-Z SDF FOLIAGE SURFELS: Pines & Marsh Brush (skip walkable timber bridges!)
                let is_bridge = c.s1 == SP_WOOD && !c.wall && c.walkable;
                if !is_bridge && (c.s1 == SP_WOOD || (c.s2 == SP_WOOD && c.s2_frac > 0.35)) {
                    let scale = 0.74 + j0 * 0.40;
                    let snowy = c.freezing || c.temp_norm < 0.088 || (gy as f32 / gh as f32) < 0.32;
                    cache.props.push(ChunkPropItem {
                        kind: MeshKind::TreeTrunk,
                        world_pos: Vec3::new(wx, h - 0.08, wz),
                        yaw: j0 * 6.28,
                        scale: Vec3::splat(scale),
                        tint: [1.0, 1.0, 1.0],
                        glow: 0.0,
                    });
                    let tint = if c.temp_norm > 0.16 {
                        [1.05, 0.82, 0.56]
                    } else if snowy {
                        [0.95, 0.99, 1.02]
                    } else {
                        [0.86 + j1 * 0.22, 0.92 + j0 * 0.14, 0.88 + j1 * 0.14]
                    };
                    cache.surfels.push(ChunkSurfelItem {
                        kind: if snowy {
                            SurfelProtoKind::SnowPine
                        } else {
                            SurfelProtoKind::PineCanopy
                        },
                        world_pos: Vec3::new(wx, h, wz),
                        yaw: j1 * 6.28,
                        scale,
                        tint,
                        wind_phase: gx as f32 * 0.37 + gy as f32 * 0.51,
                        glow_mul: 0.0,
                    });
                } else if (c.s1 == SP_PLANT || (c.s2 == SP_PLANT && c.s2_frac > 0.28)) && j0 < 0.20 {
                    let near_water = (gx as i32 - (gw as i32 / 2)).abs() < (gw as i32 / 5);
                    cache.surfels.push(ChunkSurfelItem {
                        kind: if near_water && j1 < 0.55 {
                            SurfelProtoKind::RiverReeds
                        } else {
                            SurfelProtoKind::MarshBrush
                        },
                        world_pos: Vec3::new(wx + (j0 - 0.5) * 0.4, h, wz + (j1 - 0.5) * 0.4),
                        yaw: j1 * 6.28,
                        scale: 0.64 + j0 * 0.42,
                        tint: [0.90 + j1 * 0.18, 0.96, 0.88 + j0 * 0.16],
                        wind_phase: j0 * 10.0,
                        glow_mul: 0.15,
                    });
                } else if c.s1 == SP_ROCK
                    && gx >= 5
                    && gx + 5 < gw
                    && gy >= 5
                    && gy + 5 < gh
                    && gx % 3 == 0
                    && gy % 3 == 0
                {
                    cache.props.push(ChunkPropItem {
                        kind: MeshKind::RockOutcrop,
                        world_pos: Vec3::new(wx, h - 0.85, wz),
                        yaw: j0 * 6.28,
                        scale: Vec3::new(1.08 + j0 * 0.42, 1.15 + j1 * 0.52, 1.08 + j1 * 0.42),
                        tint: [1.0, 1.0, 1.0],
                        glow: 0.0,
                    });
                }

                // LIVE ECOSYSTEM WILDLIFE HERDS + Soft Bioluminescent Marsh Wisps!
                if c.critter >= 2 && c.walkable && !c.burning && (gx % 4 == 0 && gy % 4 == 0) {
                    if j0 < 0.55 {
                        cache.props.push(ChunkPropItem {
                            kind: MeshKind::MarshDeer,
                            world_pos: Vec3::new(wx + (j0 - 0.5) * 0.6, h, wz + (j1 - 0.5) * 0.6),
                            yaw: j1 * 6.28,
                            scale: Vec3::splat(0.92 + j0 * 0.28),
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

    /// Map authoritative 3D layer altitude `gz` (`1.0=ground`, `2.0=Highland/Skybridge`) into world Y!
    pub fn sample_3d_y(&self, gx: f32, gy: f32, gz: f32) -> f32 {
        let base_y = self.sample_ground_y(gx, gy);
        let surf_z = self.surf_layer() as f32;
        if gz <= surf_z + 0.02 {
            base_y
        } else {
            let t = (gz - surf_z).clamp(0.0, 2.0);
            base_y + t * 3.55
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
}

pub fn assemble_live_frame(
    world: &LiveApproach4World,
    entities: Option<&LiveEntityState>,
    surfels_bank: &SurfelProtoBank,
    mats: &CameraMatrices,
    cam_target: Vec3,
    cam_dist: f32,
    sim_time: f32,
    layer_slice_mode: u8, // 0=ALL 4L (3D), 1=GROUND (L0-L1), 2=HIGH+SKY (L2-L3)
) -> AssembledFrame {
    let origin = mats.origin;
    let cull_rad = (cam_dist * 1.95 + 42.0).max(96.0);
    let cull_r2 = cull_rad * cull_rad;

    let mut terrain_verts = Vec::with_capacity(48_000);
    let mut terrain_idxs = Vec::with_capacity(96_000);
    let mut mesh_batches: [Vec<MeshInstance>; MESH_KIND_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut surfel_placements = Vec::with_capacity(3_072);
    let mut surfel_refs = Vec::with_capacity(80_000);
    let mut vol_splats = Vec::with_capacity(2_048);

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

    // 30 Hz Entities: Shrines, Rafts, Golden Relic, 3-Class Heroes, Projectiles!
    if let Some(ent) = entities {
        // 3 Capturable Elemental Shrines
        for s in &ent.shrines {
            let (wx, wz) = world.grid_to_world_xz(s.x, s.y);
            let gy = world.sample_ground_y(s.x, s.y);
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
            mesh_batches[MeshKind::ShrineObelisk as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, sim_time * 0.35 + s.id as f32],
                scale_pitch: [1.15, 1.15, 1.15, 0.0],
                tint_glow: [scol[0], scol[1], scol[2], 0.35],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            // Capture radius ring on ground
            mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y + 0.15, rel.z, 0.0],
                scale_pitch: [8.5, 1.0, 8.5, 0.0],
                tint_glow: [scol[0], scol[1], scol[2], 0.55],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });
            // Glowing shrine aura volumetric wisp
            vol_splats.push(VolSplat {
                pos_rad: [rel.x, rel.y + 4.3, rel.z, 1.45],
                color_alpha: [scol[0] * 1.1, scol[1] * 1.1, scol[2] * 1.1, 0.62],
                params: [1.35, 3.0, s.id as f32 * 3.1, 0.0],
            });
            push_light(Vec3::new(wx, gy + 4.0, wz), 16.0, [scol[0] * 0.9, scol[1] * 0.9, scol[2] * 0.9]);
        }

        // Carved `CellBody` Ice Rafts
        for raft in &ent.bodies {
            let (wx, wz) = world.grid_to_world_xz(raft.x, raft.y);
            let bob = 0.06 * (sim_time * 3.1 + raft.id as f32).sin();
            let rel = Vec3::new(wx, 0.04 + bob, wz) - origin;
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
        let (rx, rz) = world.grid_to_world_xz(ent.relic_x, ent.relic_y);
        let ry = world.sample_ground_y(ent.relic_x, ent.relic_y);
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

        // 3-Class Heroes & Bots (using true 3D altitude `p.z` so L2 Skybridge/Plateau heroes tower overhead!)
        for p in &ent.players {
            let (wx, wz) = world.grid_to_world_xz(p.x, p.y);
            let gy = world.sample_3d_y(p.x, p.y, p.z);
            let speed = p.vx.hypot(p.vy);
            let stride = if speed > 1.0 {
                (sim_time * 11.0 + p.id as f32 * 1.7).sin().abs() * 0.22
            } else {
                0.0
            };
            let yaw = p.aim_x.atan2(p.aim_y);
            let rel = Vec3::new(wx, gy + stride, wz) - origin;

            let kind = match p.class_id {
                CLASS_WARDEN => MeshKind::HeroKnight,
                CLASS_ALCHEMIST => MeshKind::HeroMage,
                _ => MeshKind::HeroRanger,
            };
            let team_tint = if p.team == 0 {
                [0.20, 0.58, 0.98]
            } else {
                [0.94, 0.24, 0.26]
            };

            mesh_batches[kind as usize].push(MeshInstance {
                pos_yaw: [rel.x, rel.y, rel.z, yaw],
                scale_pitch: [1.28, 1.28, 1.28, if p.dashing { 0.18 } else { 0.0 }],
                tint_glow: [
                    team_tint[0],
                    team_tint[1],
                    team_tint[2],
                    if p.dashing || p.ult_active { 0.30 } else { 0.0 },
                ],
                flags: [-1.0, 1.0, 0.0, 0.0],
            });

            // Ground ring (golden rim for self!)
            let is_self = p.id == ent.your_player_id;
            let ring_col = if is_self { [0.38, 1.0, 0.56] } else { team_tint };
            let ring_rel = Vec3::new(wx, gy + 0.05, wz) - origin;
            mesh_batches[MeshKind::GroundRing as usize].push(MeshInstance {
                pos_yaw: [ring_rel.x, ring_rel.y, ring_rel.z, 0.0],
                scale_pitch: [if is_self { 1.45 } else { 1.18 }, 1.0, if is_self { 1.45 } else { 1.18 }, 0.0],
                tint_glow: [ring_col[0], ring_col[1], ring_col[2], 0.72],
                flags: [-1.0, 0.0, 0.0, 0.0],
            });

            if p.dashing || (p.status_flags & STATUS_SHIELDED) != 0 {
                mesh_batches[MeshKind::DashHalo as usize].push(MeshInstance {
                    pos_yaw: [rel.x, rel.y, rel.z, sim_time * 5.0],
                    scale_pitch: [1.25, 1.25, 1.25, 0.0],
                    tint_glow: [0.52, 0.94, 1.0, 0.92],
                    flags: [-1.0, 0.0, 0.0, 0.0],
                });
            }

            if p.melee_active {
                mesh_batches[MeshKind::MeleeArc as usize].push(MeshInstance {
                    pos_yaw: [rel.x, rel.y, rel.z, yaw - std::f32::consts::FRAC_PI_2],
                    scale_pitch: [1.25, 1.0, 1.25, 0.0],
                    tint_glow: [1.0, 0.95, 0.65, 0.95],
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
                let bob = 0.22 * (sim_time * 3.8).sin();
                let r_rel = Vec3::new(wx, gy + 4.1 + bob, wz) - origin;
                mesh_batches[MeshKind::GoldenRelic as usize].push(MeshInstance {
                    pos_yaw: [r_rel.x, r_rel.y, r_rel.z, sim_time * 2.2],
                    scale_pitch: [0.98, 0.98, 0.98, 0.0],
                    tint_glow: [1.0, 0.94, 0.72, 0.45],
                    flags: [-1.0, 0.0, 0.0, 0.0],
                });
            }
        }

        // Elemental Projectiles + Soft Volumetric Trails (in true 3D `(x, y, z)`!)
        for pr in &ent.projectiles {
            let (wx, wz) = world.grid_to_world_xz(pr.x, pr.y);
            let gy = world.sample_3d_y(pr.x, pr.y, pr.z);
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
    }
}
