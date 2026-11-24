//! Frostmarsh Wave 12C: True 3D `Z=16..32` WebGPU Browser Client (`wasm32-unknown-unknown`).
//!
//! Powered by Smooth Surface-Nets + Client-Side WGSL 4-Style Neural Compute (`Y`-Key)
//! + Triplanar Painterly Micro-Detail + Smart 3D Cave Ceiling Cutaway (`C`-Key Capsule Walk Mode)
//! + Depth-Tested Raymarched Volumetric Fluid/Gas Overlay + Live Binary WebSocket Multiplayer
//! Receiver (`0x01` Incremental Brick3D + `0x02` Live Entities) + `Pixel-2×SS` ↔ `Clean-3D` toggle.

pub mod camera;
pub mod collision;
pub mod hero_mesh;
pub mod live_entities;
#[cfg(target_arch = "wasm32")]
pub mod renderer;
pub mod surface_nets_3d;
pub mod volume;

pub const STYLE_NAMES: [&str; 4] = [
    "NORDIC",
    "VOLCANIC EMBER",
    "ARCANE THERMAL",
    "EMERALD CAVERN",
];

pub fn style_name_for(mode: u32) -> &'static str {
    STYLE_NAMES[(mode % 4) as usize]
}

#[cfg(target_arch = "wasm32")]
mod wasm_app {
    use crate::camera::Camera3D;
    use crate::collision::{
        step_capsule_3d, CAPSULE_EYE_HEIGHT, DEFAULT_CAPSULE_HEIGHT, DEFAULT_CAPSULE_RADIUS,
    };
    use crate::hero_mesh::{
        build_archetype_meshes, build_projectile_orb_mesh, showcase_archetype_heroes, HeroInstance,
        ARCHETYPE_STRIDER, NUM_ARCHETYPES,
    };
    use crate::live_entities::{
        expand_beams_to_orbs, parse_live_entities_json, EvolutionStats, LiveBeam, TacticalEvent,
        WorldLawInfo,
    };
    use crate::renderer::WebRenderer;
    use crate::style_name_for;
    use crate::surface_nets_3d::SurfaceNetsWorld;
    use crate::volume::{BrickOccupancy3D, WorldVolume3D};
    use glam::Vec3;
    use std::collections::HashSet;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    fn perf_now_ms() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {}
                c => out.push(c),
            }
        }
        out
    }

    #[wasm_bindgen]
    pub struct FrostmarshApp {
        volume: WorldVolume3D,
        bricks: BrickOccupancy3D,
        sn_world: SurfaceNetsWorld,
        renderer: WebRenderer,
        cam: Camera3D,
        keys: HashSet<String>,
        pixel_art: bool,
        style_mode: u32,
        capsule_mode: bool,
        capsule_pos: Vec3,
        capsule_vel: Vec3,
        capsule_grounded: bool,
        capsule_cam_dist: f32,
        sim_time: f32,
        fps_smooth: f32,
        frame_ms_smooth: f32,
        volume_source: String,
        volume_ms: f64,
        sn_build_ms: f64,
        base_heroes: Vec<HeroInstance>,
        base_hero_classes: Vec<u8>,
        archetype_counts: [usize; NUM_ARCHETYPES],
        live_projectiles: Vec<HeroInstance>,
        live_beams: Vec<LiveBeam>,
        live_evolution: Option<EvolutionStats>,
        live_events: Vec<TacticalEvent>,
        live_world_law: Option<WorldLawInfo>,
        live_entities_active: bool,
        live_tick: u64,
        live_server_tps: f32,
        live_status: String,
        live_bricks_rx: u32,
        last_brick_rebuild_ms: f64,
    }

    #[wasm_bindgen]
    impl FrostmarshApp {
        /// Create and initialize the Frostmarsh WebGPU application on the given `<canvas>` id.
        /// If `fm3d_bytes` is provided, parses it as an `FM3D` volume (`Z=16..32+`); otherwise
        /// generates the built-in isotropic `256×256×16` Frostmarsh True-3D world in WASM.
        pub async fn create(
            canvas_id: String,
            fm3d_bytes: Option<js_sys::Uint8Array>,
        ) -> Result<FrostmarshApp, JsValue> {
            console_error_panic_hook::set_once();

            let window = web_sys::window().ok_or_else(|| JsValue::from_str("No browser window"))?;
            let doc = window
                .document()
                .ok_or_else(|| JsValue::from_str("No DOM document"))?;
            let el = doc
                .get_element_by_id(&canvas_id)
                .ok_or_else(|| JsValue::from_str(&format!("Canvas #{canvas_id} not found")))?;
            let canvas: web_sys::HtmlCanvasElement = el
                .dyn_into()
                .map_err(|_| JsValue::from_str("Element is not an HTMLCanvasElement"))?;

            let w = canvas.width().max(1);
            let h = canvas.height().max(1);

            let t0 = perf_now_ms();
            let (volume, volume_source) = if let Some(arr) = fm3d_bytes {
                let bytes = arr.to_vec();
                let vol = WorldVolume3D::from_fm3d_bytes(&bytes)
                    .map_err(|e| JsValue::from_str(&e))?;
                (vol, "Neural FM3D (HTTP)".to_string())
            } else {
                (
                    WorldVolume3D::generate_showcase(),
                    "Procedural Showcase (WASM)".to_string(),
                )
            };
            let volume_ms = (perf_now_ms() - t0).max(0.0);

            let bricks = volume.build_brick_occupancy();
            let t_sn = perf_now_ms();
            let sn_world = SurfaceNetsWorld::build_all(&volume);
            let sn_build_ms = (perf_now_ms() - t_sn).max(0.0);

            let archetype_meshes = build_archetype_meshes();
            let orb_mesh = build_projectile_orb_mesh();
            let showcase = showcase_archetype_heroes();
            let mut base_heroes = Vec::with_capacity(showcase.len());
            let mut base_hero_classes = Vec::with_capacity(showcase.len());
            let mut archetype_counts = [0usize; NUM_ARCHETYPES];
            let mut initial_buckets: [Vec<HeroInstance>; NUM_ARCHETYPES] =
                std::array::from_fn(|_| Vec::new());
            for (inst, cls) in showcase {
                let c = (cls as usize) % NUM_ARCHETYPES;
                base_heroes.push(inst);
                base_hero_classes.push(c as u8);
                archetype_counts[c] += 1;
                initial_buckets[c].push(inst);
            }

            let renderer = WebRenderer::from_canvas(
                canvas,
                w,
                h,
                &volume,
                &bricks,
                &sn_world,
                &archetype_meshes,
                &initial_buckets,
                &orb_mesh,
            )
            .await
            .map_err(|e| JsValue::from_str(&e))?;

            let cam = Camera3D::preset_bridge_waterfall();
            let capsule_pos = Vec3::new(113.5, 4.2, 140.0);

            Ok(FrostmarshApp {
                volume,
                bricks,
                sn_world,
                renderer,
                cam,
                keys: HashSet::new(),
                pixel_art: false,
                style_mode: 0,
                capsule_mode: false,
                capsule_pos,
                capsule_vel: Vec3::ZERO,
                capsule_grounded: false,
                capsule_cam_dist: 6.5,
                sim_time: 0.0,
                fps_smooth: 60.0,
                frame_ms_smooth: 16.6,
                volume_source,
                volume_ms,
                sn_build_ms,
                base_heroes,
                base_hero_classes,
                archetype_counts,
                live_projectiles: Vec::new(),
                live_beams: Vec::new(),
                live_evolution: None,
                live_events: Vec::new(),
                live_world_law: None,
                live_entities_active: false,
                live_tick: 0,
                live_server_tps: 0.0,
                live_status: "Offline".to_string(),
                live_bricks_rx: 0,
                last_brick_rebuild_ms: 0.0,
            })
        }

        pub fn resize(&mut self, width: u32, height: u32) {
            self.renderer.resize(width, height);
        }

        pub fn set_key(&mut self, code: &str, pressed: bool) {
            if pressed {
                self.keys.insert(code.to_string());
            } else {
                self.keys.remove(code);
            }
        }

        pub fn clear_keys(&mut self) {
            self.keys.clear();
        }

        pub fn orbit_drag(&mut self, dx: f32, dy: f32) {
            self.cam.yaw += dx * 0.0048;
            self.cam.pitch = (self.cam.pitch - dy * 0.0048).clamp(-1.45, 1.45);
        }

        pub fn zoom_wheel(&mut self, delta_y: f32) {
            if self.capsule_mode {
                // Smoothly zoom between 0.0m (first-person) and 16.0m (3rd-person cave cutaway view)
                self.capsule_cam_dist = (self.capsule_cam_dist + delta_y * 0.012).clamp(0.0, 16.0);
            } else {
                let fwd = self.cam.forward();
                let step = (-delta_y * 0.018).clamp(-6.0, 6.0);
                self.cam.pos += fwd * step;
                self.clamp_cam_bounds();
            }
        }

        pub fn set_camera_preset(&mut self, preset: u32) {
            self.capsule_mode = false;
            self.cam = match preset {
                2 => Camera3D::preset_cave_mouth(),
                _ => Camera3D::preset_bridge_waterfall(),
            };
            self.sync_entities_to_gpu();
        }

        /// Toggle stylized `Pixel-2×SS` ↔ `Clean-3D` (`T`-key). Returns `true` when `Pixel-2×SS` is active.
        pub fn toggle_pixel_style(&mut self) -> bool {
            self.pixel_art = !self.pixel_art;
            self.pixel_art
        }

        /// Cycle live client-side WGSL 4-Style Neural Compute (`Y`-key):
        /// `0 = NORDIC`, `1 = VOLCANIC EMBER`, `2 = ARCANE THERMAL`, `3 = EMERALD CAVERN`.
        pub fn cycle_style_mode(&mut self) -> u32 {
            self.style_mode = (self.style_mode + 1) % 4;
            self.style_mode
        }

        pub fn set_style_mode(&mut self, mode: u32) -> u32 {
            self.style_mode = mode % 4;
            self.style_mode
        }

        pub fn get_style_name(&self) -> String {
            style_name_for(self.style_mode).to_string()
        }

        /// Toggle 3D Capsule Collision Walk Mode (`C`-key) with Smart 3D Cave Ceiling Cutaway.
        /// Returns `true` when Capsule Walk is active.
        pub fn toggle_capsule_mode(&mut self) -> bool {
            self.capsule_mode = !self.capsule_mode;
            if self.capsule_mode {
                let foot_y = (self.cam.pos.y - CAPSULE_EYE_HEIGHT)
                    .clamp(1.2, self.volume.depth as f32 - DEFAULT_CAPSULE_HEIGHT - 0.2);
                self.capsule_pos = Vec3::new(
                    self.cam.pos.x.clamp(2.0, self.volume.width as f32 - 2.0),
                    foot_y,
                    self.cam.pos.z.clamp(2.0, self.volume.height as f32 - 2.0),
                );
                self.capsule_vel = Vec3::ZERO;
                let res = step_capsule_3d(
                    &self.volume,
                    self.capsule_pos,
                    self.capsule_vel,
                    DEFAULT_CAPSULE_RADIUS,
                    DEFAULT_CAPSULE_HEIGHT,
                    1.0 / 60.0,
                );
                self.capsule_pos = res.pos;
                self.capsule_vel = res.vel;
                self.capsule_grounded = res.grounded;
                self.update_capsule_camera();
            }
            self.sync_entities_to_gpu();
            self.capsule_mode
        }

        fn update_capsule_camera(&mut self) {
            let eye = self.capsule_pos + Vec3::new(0.0, CAPSULE_EYE_HEIGHT, 0.0);
            if self.capsule_cam_dist <= 0.2 {
                self.cam.pos = eye;
            } else {
                let fwd = self.cam.forward();
                let lift = (self.capsule_cam_dist * 0.28).min(4.2);
                self.cam.pos = eye - fwd * self.capsule_cam_dist + Vec3::new(0.0, lift, 0.0);
                self.cam.pos.y = self.cam.pos.y.max(1.0);
            }
        }

        fn build_orb_instances(&self) -> Vec<HeroInstance> {
            let mut orbs = Vec::with_capacity(
                self.live_projectiles.len() + self.live_beams.len() * 20,
            );
            orbs.extend_from_slice(&self.live_projectiles);
            if !self.live_beams.is_empty() {
                expand_beams_to_orbs(&self.live_beams, self.sim_time, &mut orbs);
            }
            orbs
        }

        fn sync_entities_to_gpu(&mut self) {
            let mut buckets: [Vec<HeroInstance>; NUM_ARCHETYPES] =
                std::array::from_fn(|_| Vec::new());
            for (idx, inst) in self.base_heroes.iter().enumerate() {
                let cls = self
                    .base_hero_classes
                    .get(idx)
                    .copied()
                    .unwrap_or(ARCHETYPE_STRIDER) as usize
                    % NUM_ARCHETYPES;
                buckets[cls].push(*inst);
            }
            if self.capsule_mode && self.capsule_cam_dist > 0.6 {
                // Render the player's own Strider Hero Knight figurine inside Capsule Walk Mode
                buckets[ARCHETYPE_STRIDER as usize].push(HeroInstance {
                    pos_yaw: [
                        self.capsule_pos.x,
                        self.capsule_pos.y,
                        self.capsule_pos.z,
                        self.cam.yaw - std::f32::consts::FRAC_PI_2,
                    ],
                    faction_scale: [0.34, 0.86, 1.00, 1.0],
                });
            }
            let orbs = self.build_orb_instances();
            self.renderer.update_entities(&buckets, &orbs);
        }

        /// Hot-swap to an external `FM3D` neural volume (`Z=16..32+`) in browser.
        pub fn load_fm3d_volume(
            &mut self,
            bytes: js_sys::Uint8Array,
            label: String,
        ) -> Result<(), JsValue> {
            let raw = bytes.to_vec();
            let t0 = perf_now_ms();
            let volume =
                WorldVolume3D::from_fm3d_bytes(&raw).map_err(|e| JsValue::from_str(&e))?;
            self.volume_ms = (perf_now_ms() - t0).max(0.0);
            let bricks = volume.build_brick_occupancy();
            let t_sn = perf_now_ms();
            let sn_world = SurfaceNetsWorld::build_all(&volume);
            self.sn_build_ms = (perf_now_ms() - t_sn).max(0.0);
            self.renderer.upload_volume(&volume, &bricks, &sn_world);
            self.volume = volume;
            self.bricks = bricks;
            self.sn_world = sn_world;
            self.volume_source = label;
            Ok(())
        }

        /// Hot-swap back to the built-in procedural isotropic `256×256×16` showcase world.
        pub fn load_procedural_showcase(&mut self) {
            self.load_procedural_showcase_depth(16);
        }

        /// Generate and hot-swap a procedural isotropic `256×256×D` (`D = 16..=32`) showcase world.
        pub fn load_procedural_showcase_depth(&mut self, depth: u32) {
            let d = (depth as usize).clamp(16, 32);
            let t0 = perf_now_ms();
            let volume = WorldVolume3D::generate_showcase_with_depth(d);
            self.volume_ms = (perf_now_ms() - t0).max(0.0);
            let bricks = volume.build_brick_occupancy();
            let t_sn = perf_now_ms();
            let sn_world = SurfaceNetsWorld::build_all(&volume);
            self.sn_build_ms = (perf_now_ms() - t_sn).max(0.0);
            self.renderer.upload_volume(&volume, &bricks, &sn_world);
            self.volume = volume;
            self.bricks = bricks;
            self.sn_world = sn_world;
            self.volume_source = format!("Procedural Showcase Z={d} (WASM)");
        }

        /// Apply a live binary `0x01` `16×16×D` (`D = 16..=32`) volumetric brick from WebSocket!
        /// Performs incremental Surface-Nets brick rebuild + GPU `queue.write_texture` sub-brick upload.
        /// Returns elapsed rebuild+upload time in milliseconds.
        pub fn apply_brick3d_raw(
            &mut self,
            bx: u16,
            by: u16,
            depth: u8,
            density_rgba: &[u8],
            albedo_rgba: &[u8],
        ) -> Result<f64, JsValue> {
            let t0 = perf_now_ms();
            let patch = self
                .volume
                .apply_brick3d_raw(
                    bx as usize,
                    by as usize,
                    depth as usize,
                    density_rgba,
                    albedo_rgba,
                )
                .map_err(|e| JsValue::from_str(&e))?;

            if patch.resized_depth {
                self.bricks = self.volume.build_brick_occupancy();
                self.sn_world = SurfaceNetsWorld::build_all(&self.volume);
                self.renderer
                    .upload_volume(&self.volume, &self.bricks, &self.sn_world);
            } else {
                let occ_sub = self.volume.update_brick_occupancy_region(
                    &mut self.bricks,
                    bx as usize,
                    by as usize,
                );
                self.sn_world
                    .rebuild_brick_and_neighbors(&self.volume, bx as usize, by as usize);
                self.renderer.upload_brick_subregion(
                    &self.volume,
                    &self.bricks,
                    &self.sn_world,
                    patch,
                    occ_sub,
                );
            }

            let elapsed_ms = (perf_now_ms() - t0).max(0.0);
            self.live_bricks_rx = self.live_bricks_rx.saturating_add(1);
            self.last_brick_rebuild_ms = elapsed_ms;
            self.volume_source = format!("Live WS Stream (Z={})", self.volume.depth);
            Ok(elapsed_ms)
        }

        /// Apply a live `0x02` multiplayer JSON entity, actuator beam, & evolution frame from WebSocket!
        /// Updates live 5-archetype hero figurines, glowing projectiles, diagonal beam orb-streams,
        /// world evolution stats, and telemetry HUD.
        /// Returns total active rendered entities (`players + projectiles + beams`).
        pub fn apply_live_entities_json(&mut self, json_str: &str) -> Result<u32, JsValue> {
            let parsed =
                parse_live_entities_json(json_str).map_err(|e| JsValue::from_str(&e))?;
            self.base_heroes = parsed.heroes;
            self.base_hero_classes = parsed.hero_classes;
            self.archetype_counts = parsed.archetype_counts;
            self.live_projectiles = parsed.projectiles;
            self.live_beams = parsed.beams;
            if parsed.evolution.is_some() {
                self.live_evolution = parsed.evolution;
            }
            if parsed.has_events {
                self.live_events = parsed.events;
            }
            if parsed.world_law.is_some() {
                self.live_world_law = parsed.world_law;
            }
            self.live_entities_active = true;
            self.live_tick = parsed.tick;
            self.live_server_tps = parsed.server_tps;
            if let Some(st) = parsed.status {
                self.live_status = st;
            } else {
                self.live_status = "Live WS".to_string();
            }
            self.sync_entities_to_gpu();
            Ok((self.base_heroes.len() + self.live_projectiles.len() + self.live_beams.len())
                as u32)
        }

        pub fn tick_and_render(&mut self, dt_sec: f32) {
            let dt = dt_sec.clamp(0.001, 0.1);
            self.sim_time += dt;

            let inst_ms = dt * 1000.0;
            let inst_fps = 1.0 / dt;
            self.frame_ms_smooth = self.frame_ms_smooth * 0.90 + inst_ms * 0.10;
            self.fps_smooth = self.fps_smooth * 0.90 + inst_fps * 0.10;

            let sprint =
                self.keys.contains("ShiftLeft") || self.keys.contains("ShiftRight");

            if self.capsule_mode {
                let walk_speed = if sprint { 9.5 } else { 5.2 };
                let fwd = self.cam.flat_forward();
                let right = self.cam.flat_right();
                let mut wish = Vec3::ZERO;
                if self.keys.contains("KeyW") || self.keys.contains("ArrowUp") {
                    wish += fwd;
                }
                if self.keys.contains("KeyS") || self.keys.contains("ArrowDown") {
                    wish -= fwd;
                }
                if self.keys.contains("KeyD") || self.keys.contains("ArrowRight") {
                    wish += right;
                }
                if self.keys.contains("KeyA") || self.keys.contains("ArrowLeft") {
                    wish -= right;
                }
                let wish_dir = if wish.length_squared() > 1e-4 {
                    wish.normalize()
                } else {
                    Vec3::ZERO
                };
                self.capsule_vel.x = wish_dir.x * walk_speed;
                self.capsule_vel.z = wish_dir.z * walk_speed;
                if self.capsule_grounded && self.keys.contains("Space") {
                    self.capsule_vel.y = 7.4;
                }

                let res = step_capsule_3d(
                    &self.volume,
                    self.capsule_pos,
                    self.capsule_vel,
                    DEFAULT_CAPSULE_RADIUS,
                    DEFAULT_CAPSULE_HEIGHT,
                    dt,
                );
                self.capsule_pos = res.pos;
                self.capsule_vel = res.vel;
                self.capsule_grounded = res.grounded;
                self.update_capsule_camera();
                self.sync_entities_to_gpu();
            } else {
                let fly_speed = if sprint { 24.0 } else { 11.5 };
                let fwd = self.cam.forward();
                let right = self.cam.right();
                if self.keys.contains("KeyW") || self.keys.contains("ArrowUp") {
                    self.cam.pos += fwd * fly_speed * dt;
                }
                if self.keys.contains("KeyS") || self.keys.contains("ArrowDown") {
                    self.cam.pos -= fwd * fly_speed * dt;
                }
                if self.keys.contains("KeyD") || self.keys.contains("ArrowRight") {
                    self.cam.pos += right * fly_speed * dt;
                }
                if self.keys.contains("KeyA") || self.keys.contains("ArrowLeft") {
                    self.cam.pos -= right * fly_speed * dt;
                }
                if self.keys.contains("KeyE") || self.keys.contains("Space") {
                    self.cam.pos.y += fly_speed * dt;
                }
                if self.keys.contains("KeyQ") {
                    self.cam.pos.y -= fly_speed * dt;
                }
                self.clamp_cam_bounds();
                if !self.live_beams.is_empty() {
                    let orbs = self.build_orb_instances();
                    self.renderer.update_orb_instances(&orbs);
                }
            }

            let player_cutaway = [
                self.capsule_pos.x,
                self.capsule_pos.y,
                self.capsule_pos.z,
                if self.capsule_mode { 1.0 } else { 0.0 },
            ];

            self.renderer.render_frame(
                &self.cam,
                self.sim_time,
                self.pixel_art,
                self.style_mode,
                player_cutaway,
            );
        }

        fn clamp_cam_bounds(&mut self) {
            self.cam.pos.x = self.cam.pos.x.clamp(-20.0, self.volume.width as f32 + 20.0);
            self.cam.pos.y = self.cam.pos.y.clamp(0.6, self.volume.depth as f32 + 48.0);
            self.cam.pos.z = self.cam.pos.z.clamp(-20.0, self.volume.height as f32 + 20.0);
        }

        /// Return compact JSON telemetry for the browser HUD overlay.
        pub fn get_hud_stats(&self) -> String {
            let voxels = self.volume.width * self.volume.height * self.volume.depth;
            let tris = self.renderer.sn_index_count / 3;
            let style = if self.pixel_art {
                "Pixel-2xSS"
            } else {
                "Clean-3D"
            };
            let walk = if self.capsule_mode {
                "Capsule-3D"
            } else {
                "Free-Fly"
            };
            let archetype_types = self.archetype_counts.iter().filter(|&&c| c > 0).count();
            let evo = self.live_evolution.unwrap_or_default();
            let mut events_json = String::from("[");
            for (idx, ev) in self.live_events.iter().enumerate() {
                if idx > 0 {
                    events_json.push(',');
                }
                events_json.push_str("{\"label\":\"");
                events_json.push_str(&json_escape(&ev.label));
                events_json.push('"');
                if let Some(team) = ev.team {
                    events_json.push_str(&format!(",\"team\":{team}"));
                }
                if let Some(tick) = ev.tick {
                    events_json.push_str(&format!(",\"tick\":{tick}"));
                }
                events_json.push('}');
            }
            events_json.push(']');
            let (wl_id, wl_name, wl_g, wl_dt, wl_precip) = match &self.live_world_law {
                Some(wl) => (
                    wl.id,
                    wl.name.as_str(),
                    wl.gravity_g,
                    wl.ambient_shift,
                    wl.precipitation,
                ),
                None => (0u8, "STANDARD", 1.0f32, 0.0f32, 1.0f32),
            };
            format!(
                concat!(
                    "{{",
                    "\"fps\":{:.1},",
                    "\"frame_ms\":{:.2},",
                    "\"adapter\":\"{}\",",
                    "\"backend\":\"{}\",",
                    "\"triangles\":{},",
                    "\"vertices\":{},",
                    "\"vol_w\":{},",
                    "\"vol_h\":{},",
                    "\"vol_d\":{},",
                    "\"voxels\":{},",
                    "\"non_empty_bricks\":{},",
                    "\"total_bricks\":{},",
                    "\"style\":\"{}\",",
                    "\"style_mode\":{},",
                    "\"neural_style\":\"{}\",",
                    "\"walk_mode\":\"{}\",",
                    "\"grounded\":{},",
                    "\"cutaway_active\":{},",
                    "\"cam_x\":{:.1},",
                    "\"cam_y\":{:.1},",
                    "\"cam_z\":{:.1},",
                    "\"volume_source\":\"{}\",",
                    "\"volume_ms\":{:.1},",
                    "\"sn_build_ms\":{:.1},",
                    "\"live_active\":{},",
                    "\"live_players\":{},",
                    "\"live_projectiles\":{},",
                    "\"live_beams\":{},",
                    "\"archetype_types\":{},",
                    "\"archetype_counts\":[{},{},{},{},{}],",
                    "\"evo_active\":{},",
                    "\"evo_epoch\":{},",
                    "\"evo_basalt\":{},",
                    "\"evo_trees\":{},",
                    "\"evo_glaciers\":{},",
                    "\"evo_erosion\":{},",
                    "\"events\":{},",
                    "\"world_law_id\":{},",
                    "\"world_law_name\":\"{}\",",
                    "\"world_law_g\":{:.2},",
                    "\"world_law_dt\":{:.1},",
                    "\"world_law_precip\":{:.2},",
                    "\"live_bricks_rx\":{},",
                    "\"last_brick_ms\":{:.3},",
                    "\"live_tick\":{},",
                    "\"live_server_tps\":{:.1},",
                    "\"live_status\":\"{}\"",
                    "}}"
                ),
                self.fps_smooth,
                self.frame_ms_smooth,
                json_escape(&self.renderer.adapter_name),
                json_escape(&self.renderer.adapter_backend),
                tris,
                self.renderer.sn_vertex_count,
                self.volume.width,
                self.volume.height,
                self.volume.depth,
                voxels,
                self.bricks.non_empty_bricks,
                self.bricks.total_bricks,
                style,
                self.style_mode,
                style_name_for(self.style_mode),
                walk,
                self.capsule_grounded,
                self.capsule_mode,
                self.cam.pos.x,
                self.cam.pos.y,
                self.cam.pos.z,
                json_escape(&self.volume_source),
                self.volume_ms,
                self.sn_build_ms,
                self.live_entities_active,
                self.base_heroes.len(),
                self.live_projectiles.len(),
                self.live_beams.len(),
                archetype_types,
                self.archetype_counts[0],
                self.archetype_counts[1],
                self.archetype_counts[2],
                self.archetype_counts[3],
                self.archetype_counts[4],
                self.live_evolution.is_some(),
                evo.epoch,
                evo.basalt,
                evo.trees,
                evo.glaciers,
                evo.erosion,
                events_json,
                wl_id,
                json_escape(wl_name),
                wl_g,
                wl_dt,
                wl_precip,
                self.live_bricks_rx,
                self.last_brick_rebuild_ms,
                self.live_tick,
                self.live_server_tps,
                json_escape(&self.live_status),
            )
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_app::FrostmarshApp;

#[cfg(test)]
mod tests {
    use crate::collision::{step_capsule_3d, DEFAULT_CAPSULE_HEIGHT, DEFAULT_CAPSULE_RADIUS};
    use crate::hero_mesh::{
        archetype_name, build_archetype_meshes, build_hero_knight_mesh, build_projectile_orb_mesh,
        showcase_archetype_heroes, showcase_hero_instances, ARCHETYPE_BURROWER,
        ARCHETYPE_EMBERSTALKER, ARCHETYPE_SKYWING, ARCHETYPE_STRIDER, ARCHETYPE_TIDEKIN,
        NUM_ARCHETYPES,
    };
    use crate::live_entities::{
        expand_beams_to_orbs, parse_live_entities_json, EvolutionStats, WorldLawInfo,
    };
    use crate::style_name_for;
    use crate::surface_nets_3d::{extract_brick, SurfaceNetsWorld};
    use crate::volume::WorldVolume3D;
    use glam::Vec3;

    #[test]
    fn test_showcase_surface_nets_and_capsule() {
        let vol = WorldVolume3D::generate_showcase();
        assert_eq!((vol.width, vol.height, vol.depth), (256, 256, 16));
        let bricks = vol.build_brick_occupancy();
        assert!(bricks.non_empty_bricks > 0 && bricks.non_empty_bricks <= bricks.total_bricks);

        let sn = SurfaceNetsWorld::build_all(&vol);
        assert!(sn.total_triangles() > 50_000);
        let (verts, idxs) = sn.flatten_buffers();
        assert_eq!(verts.len(), sn.total_vertices());
        assert_eq!(idxs.len(), sn.total_triangles() * 3);

        let hero = build_hero_knight_mesh();
        assert!(!hero.vertices.is_empty());
        let orb = build_projectile_orb_mesh();
        assert_eq!(orb.indices.len(), 24);
        assert_eq!(showcase_hero_instances().len(), 4);

        let res = step_capsule_3d(
            &vol,
            Vec3::new(126.5, 9.2, 128.0),
            Vec3::ZERO,
            DEFAULT_CAPSULE_RADIUS,
            DEFAULT_CAPSULE_HEIGHT,
            1.0 / 60.0,
        );
        assert!(res.max_penetration < 0.05);
    }

    #[test]
    fn test_arbitrary_depth_z16_to_z32_and_incremental_brick_patch() {
        // Test Z=24 and Z=32 volumes work seamlessly with no hardcoded 16
        for depth in [16usize, 24, 32] {
            let mut vol = WorldVolume3D::new_empty(64, depth, 64);
            // Build a simple horizontal floor at y = 4.0 and high cliff at y = depth - 4
            for iz in 0..vol.height {
                for ix in 0..vol.width {
                    for iy in 0..vol.depth {
                        let solid = if iy < 4 || (ix > 40 && iy < depth - 4) {
                            255u8
                        } else {
                            0u8
                        };
                        let idx = vol.byte_index(ix, iy, iz);
                        vol.density_mat[idx] = solid;
                        vol.density_mat[idx + 3] = 200;
                        vol.albedo_thermal[idx] = 120;
                        vol.albedo_thermal[idx + 1] = 130;
                        vol.albedo_thermal[idx + 2] = 140;
                    }
                }
            }

            let mut occ = vol.build_brick_occupancy();
            assert_eq!(occ.by_count, depth.div_ceil(8));
            let mut sn = SurfaceNetsWorld::build_all(&vol);
            assert!(sn.total_triangles() > 0);

            // Patch brick (1, 1) with a glowing magma pillar up to depth - 2
            let n = depth * 16 * 16 * 4;
            let mut den = vec![0u8; n];
            let mut alb = vec![0u8; n];
            for alt in 0..(depth - 2) {
                for dy in 4..12 {
                    for dx in 4..12 {
                        let off = ((alt * 16 + dy) * 16 + dx) * 4;
                        den[off] = 255;
                        den[off + 3] = 90;
                        alb[off] = 255;
                        alb[off + 1] = 120;
                        alb[off + 2] = 30;
                        alb[off + 3] = 240;
                    }
                }
            }
            let patch = vol
                .apply_brick3d_raw(1, 1, depth, &den, &alb)
                .expect("apply_brick3d_raw");
            assert!(!patch.resized_depth);
            assert_eq!((patch.cw, patch.ch, patch.depth), (16, 16, depth));

            let occ_sub = vol.update_brick_occupancy_region(&mut occ, 1, 1);
            assert!(occ_sub.cw > 0 && occ_sub.ch > 0);
            let rebuilt = sn.rebuild_brick_and_neighbors(&vol, 1, 1);
            assert_eq!(rebuilt, 9);

            // Single-brick extraction check
            let single = extract_brick(&vol, 1, 1);
            assert!(!single.indices.is_empty());

            let (sub_dm, sub_at) = vol.pack_brick_subregion(patch.x0, patch.z0, patch.cw, patch.ch);
            assert_eq!(sub_dm.len(), n);
            assert_eq!(sub_at.len(), n);
        }

        // Test live Z=16 -> Z=32 depth upgrade via apply_brick3d_raw
        let mut vol = WorldVolume3D::new_empty(32, 16, 32);
        let n32 = 32 * 16 * 16 * 4;
        let den32 = vec![200u8; n32];
        let alb32 = vec![150u8; n32];
        let patch = vol
            .apply_brick3d_raw(0, 0, 32, &den32, &alb32)
            .expect("live depth upgrade to Z=32");
        assert!(patch.resized_depth);
        assert_eq!(vol.depth, 32);
        assert_eq!(vol.density_mat.len(), 32 * 32 * 32 * 4);
    }

    #[test]
    fn test_live_entities_json_parser_and_styles() {
        assert_eq!(style_name_for(0), "NORDIC");
        assert_eq!(style_name_for(1), "VOLCANIC EMBER");
        assert_eq!(style_name_for(2), "ARCANE THERMAL");
        assert_eq!(style_name_for(3), "EMERALD CAVERN");
        assert_eq!(style_name_for(4), "NORDIC");

        let json = r#"{
            "tick": 420,
            "world_day": 3,
            "players": [
                {"id": 1, "x": 126.5, "y": 9.04, "z": 128.8, "yaw": 1.15, "color": [0.16, 0.52, 0.98], "scale": 1.0},
                {"id": 2, "gx": 113.5, "gy": 128.0, "gz": 4.02, "facing": -0.5, "faction": 1}
            ],
            "projectiles": [
                {"id": 99, "x": 128.0, "y": 11.2, "z": 126.0, "color": [1.0, 0.45, 0.12], "radius": 0.42}
            ],
            "telemetry": {
                "server_tps": 29.8,
                "online_players": 2,
                "status": "Live WS Demo"
            }
        }"#;
        let parsed = parse_live_entities_json(json).expect("parse live entities");
        assert_eq!(parsed.tick, 420);
        assert_eq!(parsed.world_day, 3);
        assert_eq!(parsed.heroes.len(), 2);
        assert_eq!(parsed.projectiles.len(), 1);
        assert!((parsed.heroes[1].pos_yaw[1] - 4.02).abs() < 1e-4);
        assert!((parsed.projectiles[0].faction_scale[3] - 0.42).abs() < 1e-4);
        assert!((parsed.server_tps - 29.8).abs() < 1e-3);
        assert_eq!(parsed.status.as_deref(), Some("Live WS Demo"));
        assert_eq!(parsed.hero_classes, vec![0, 0]);
        assert!(parsed.beams.is_empty());
        assert!(parsed.evolution.is_none());
    }

    #[test]
    fn test_wave14d_five_archetypes_beams_and_evolution() {
        // 1. All 5 distinct 3D archetype silhouettes compile and have expected animated traits
        let meshes = build_archetype_meshes();
        assert_eq!(meshes.len(), NUM_ARCHETYPES);
        for (cid, mesh) in meshes.iter().enumerate() {
            assert!(!mesh.vertices.is_empty(), "archetype {cid} has vertices");
            assert!(!mesh.indices.is_empty(), "archetype {cid} has indices");
        }
        assert_eq!(archetype_name(ARCHETYPE_STRIDER), "Strider");
        assert_eq!(archetype_name(ARCHETYPE_SKYWING), "Skywing");
        assert_eq!(archetype_name(ARCHETYPE_TIDEKIN), "Tidekin");
        assert_eq!(archetype_name(ARCHETYPE_EMBERSTALKER), "Emberstalker");
        assert_eq!(archetype_name(ARCHETYPE_BURROWER), "Burrower");

        // Skywing must have wide soaring wingspan (> 3.5m total) and GPU wing-flap weights
        let sky = &meshes[ARCHETYPE_SKYWING as usize];
        let max_abs_x = sky
            .vertices
            .iter()
            .map(|v| v.pos[0].abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_x > 1.8, "Skywing half-wingspan should exceed 1.8m, got {max_abs_x}");
        assert!(
            sky.vertices.iter().any(|v| v.flags[2] >= 0.9),
            "Skywing must encode wing-flap weight in flags.z"
        );

        // Burrower must encode spinning crystal-amber bore drill (flags.w >= 1.5)
        let bur = &meshes[ARCHETYPE_BURROWER as usize];
        assert!(
            bur.vertices.iter().any(|v| v.flags[3] >= 1.5 && v.flags[1] > 0.5),
            "Burrower must have glowing spinning crystal-amber drill vertices"
        );

        // Default showcase features all 5 archetypes
        let showcase = showcase_archetype_heroes();
        assert_eq!(showcase.len(), 5);
        for expected_cls in 0..5u8 {
            assert!(
                showcase.iter().any(|(_, cls)| *cls == expected_cls),
                "showcase missing archetype {expected_cls}"
            );
        }

        // 2. Parse full Wave 14D JSON frame with class_id (0..4), diagonal beams, and evolution stats
        let json = r#"{
            "tick": 1024,
            "world_day": 5,
            "players": [
                {"id": 1, "class_id": 0, "x": 126.5, "y": 9.04, "z": 128.8, "yaw": 1.15},
                {"id": 2, "class_id": 1, "x": 120.0, "y": 14.5, "z": 125.0, "yaw": 0.4, "scale": 1.1},
                {"id": 3, "class_id": 2, "x": 115.0, "y": 3.15, "z": 129.0, "yaw": -0.2},
                {"id": 4, "class_id": 3, "x": 100.2, "y": 4.25, "z": 151.4, "yaw": -1.9},
                {"id": 5, "class_id": 4, "x": 93.2, "y": 4.25, "z": 153.6, "yaw": -1.7, "color": [1.0, 0.7, 0.15]}
            ],
            "projectiles": [
                {"id": 50, "x": 125.0, "y": 11.0, "z": 127.0, "color": [0.2, 0.9, 1.0], "radius": 0.35}
            ],
            "beams": [
                {"x0": 120.0, "y0": 14.5, "z0": 125.0, "x1": 128.0, "y1": 9.1, "z1": 128.5, "color": [1.0, 0.56, 0.14]},
                {"x0": 115.0, "y0": 4.0, "z0": 129.0, "x1": 102.0, "y1": 6.5, "z1": 148.0, "color": [0.16, 0.92, 1.0], "radius": 0.30}
            ],
            "evolution": {
                "epoch": 42,
                "basalt": 128,
                "trees": 64,
                "glaciers": 19,
                "erosion": 310
            },
            "telemetry": {
                "server_tps": 30.0,
                "online_players": 5,
                "status": "Nordic Viaduct Z=32"
            }
        }"#;

        let parsed = parse_live_entities_json(json).expect("parse Wave 14D payload");
        assert_eq!(parsed.tick, 1024);
        assert_eq!(parsed.heroes.len(), 5);
        assert_eq!(parsed.hero_classes, vec![0, 1, 2, 3, 4]);
        assert_eq!(parsed.archetype_counts, [1, 1, 1, 1, 1]);
        assert!((parsed.heroes[1].pos_yaw[1] - 14.5).abs() < 1e-4);
        assert_eq!(parsed.beams.len(), 2);
        assert!((parsed.beams[0].start[1] - 14.5).abs() < 1e-4);
        assert!((parsed.beams[0].end[1] - 9.1).abs() < 1e-4);
        assert!((parsed.beams[1].radius - 0.30).abs() < 1e-4);
        assert_eq!(
            parsed.evolution,
            Some(EvolutionStats {
                epoch: 42,
                basalt: 128,
                trees: 64,
                glaciers: 19,
                erosion: 310,
            })
        );

        // 3. Diagonal beams expand into glowing 3D volumetric orb-streams spanning start -> end
        let mut beam_orbs = Vec::new();
        expand_beams_to_orbs(&parsed.beams, 1.25, &mut beam_orbs);
        assert!(beam_orbs.len() >= 20, "expected dense volumetric orb-stream, got {}", beam_orbs.len());
        let first = beam_orbs[0].pos_yaw;
        assert!((first[0] - 120.0).abs() < 1e-3 && (first[1] - 14.5).abs() < 1e-3);
    }

    #[test]
    fn test_wave18a_tactical_events_parsing() {
        // String-array format (standalone demo / compact feed)
        let str_json = r#"{
            "tick": 200,
            "players": [],
            "events": [
                "TIDEKIN QUENCH RESCUE!",
                "SUBTERRANEAN BREACH!",
                "SHIELD WALL!",
                "CRYO-BRIDGE FORGED!"
            ]
        }"#;
        let parsed_str = parse_live_entities_json(str_json).expect("parse string events");
        assert!(parsed_str.has_events);
        assert_eq!(parsed_str.events.len(), 4);
        assert_eq!(parsed_str.events[0].label, "TIDEKIN QUENCH RESCUE!");
        assert_eq!(parsed_str.events[1].label, "SUBTERRANEAN BREACH!");
        assert_eq!(parsed_str.events[2].label, "SHIELD WALL!");
        assert_eq!(parsed_str.events[3].label, "CRYO-BRIDGE FORGED!");
        assert_eq!(parsed_str.events[0].team, None);

        // Structured object-array format (live Rust frostmarsh-server 0x02 bridge)
        let obj_json = r#"{
            "tick": 512,
            "players": [{"id": 1, "class_id": 2, "x": 115.0, "y": 3.2, "z": 128.0, "team": 0}],
            "events": [
                {"label": "TIDEKIN QUENCH RESCUE!", "team": 0, "x": 115.0, "y": 128.0, "tick": 500},
                {"label": "SUBTERRANEAN BREACH!", "team": 1, "x": 96.0, "y": 152.0, "tick": 510}
            ]
        }"#;
        let parsed_obj = parse_live_entities_json(obj_json).expect("parse object events");
        assert!(parsed_obj.has_events);
        assert_eq!(parsed_obj.events.len(), 2);
        assert_eq!(parsed_obj.events[0].label, "TIDEKIN QUENCH RESCUE!");
        assert_eq!(parsed_obj.events[0].team, Some(0));
        assert_eq!(parsed_obj.events[0].tick, Some(500));
        assert_eq!(parsed_obj.events[1].label, "SUBTERRANEAN BREACH!");
        assert_eq!(parsed_obj.events[1].team, Some(1));
        assert_eq!(parsed_obj.events[1].tick, Some(510));
        assert!(parsed_obj.world_law.is_none());

        // Additive Wave 20C `"world_law"` object parsing
        let law_json = r#"{
            "tick": 777,
            "players": [],
            "world_law": {
                "id": 3,
                "name": "LOW-G MAGMA",
                "gravity_g": 0.35,
                "ambient_shift": 5500.0,
                "precipitation": 0.25
            }
        }"#;
        let parsed_law = parse_live_entities_json(law_json).expect("parse world_law JSON");
        assert_eq!(
            parsed_law.world_law,
            Some(WorldLawInfo {
                id: 3,
                name: "LOW-G MAGMA".to_string(),
                gravity_g: 0.35,
                ambient_shift: 5500.0,
                precipitation: 0.25,
            })
        );
    }

    #[test]
    fn test_fm3d_memory_parser() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FM3D");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());
        let n = 4 * 4 * 32 * 4;
        buf.extend(vec![128u8; n]);
        buf.extend(vec![200u8; n]);
        let vol = WorldVolume3D::from_fm3d_bytes(&buf).expect("parse FM3D Z=32");
        assert_eq!((vol.width, vol.height, vol.depth), (4, 4, 32));
        assert_eq!(vol.density_mat[0], 128);
        assert_eq!(vol.albedo_thermal[0], 200);
    }

    #[test]
    fn test_all_wgsl_shaders_compile_cleanly() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = futures_lite_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("GPU adapter for WGSL validation");
        let (device, _queue) = futures_lite_block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgsl_validate_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .expect("GPU device for WGSL validation");

        let shaders = [
            ("field_mesh.wgsl", include_str!("../shaders/field_mesh.wgsl")),
            ("hero_mesh.wgsl", include_str!("../shaders/hero_mesh.wgsl")),
            (
                "volumetric_overlay.wgsl",
                include_str!("../shaders/volumetric_overlay.wgsl"),
            ),
            ("post_resolve.wgsl", include_str!("../shaders/post_resolve.wgsl")),
        ];
        for (name, src) in shaders {
            let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
            });
        }
    }

    fn futures_lite_block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::task::{Context, Poll, Wake, Waker};
        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = Waker::from(std::sync::Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::yield_now();
        }
    }
}
