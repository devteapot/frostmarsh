//! Frostmarsh Iteration 9C: True 3D `Z=16` WebGPU Browser Client (`wasm32-unknown-unknown`).
//!
//! Powered by Smooth Surface-Nets + Direct 3D-Texture Shading + Raymarched Volumetric Fluid/Gas
//! Overlay + 3D Hero Knights + Continuous `∇solid_density` 3D Capsule Walk Mode + `Pixel-2×SS`
//! ↔ `Clean-3D` live toggle.

pub mod camera;
pub mod collision;
pub mod hero_mesh;
#[cfg(target_arch = "wasm32")]
pub mod renderer;
pub mod surface_nets_3d;
pub mod volume;

#[cfg(target_arch = "wasm32")]
mod wasm_app {
    use crate::camera::Camera3D;
    use crate::collision::{
        step_capsule_3d, CAPSULE_EYE_HEIGHT, DEFAULT_CAPSULE_HEIGHT, DEFAULT_CAPSULE_RADIUS,
    };
    use crate::hero_mesh::{build_hero_knight_mesh, showcase_hero_instances};
    use crate::renderer::WebRenderer;
    use crate::surface_nets_3d::SurfaceNetsWorld;
    use crate::volume::WorldVolume3D;
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
        renderer: WebRenderer,
        cam: Camera3D,
        keys: HashSet<String>,
        pixel_art: bool,
        capsule_mode: bool,
        capsule_pos: Vec3,
        capsule_vel: Vec3,
        capsule_grounded: bool,
        sim_time: f32,
        fps_smooth: f32,
        frame_ms_smooth: f32,
        volume_source: String,
        volume_ms: f64,
        sn_build_ms: f64,
        non_empty_bricks: usize,
        total_bricks: usize,
    }

    #[wasm_bindgen]
    impl FrostmarshApp {
        /// Create and initialize the Frostmarsh WebGPU application on the given `<canvas>` id.
        /// If `fm3d_bytes` is provided, parses it as an `FM3D` volume; otherwise generates the
        /// built-in isotropic `256×256×16` Frostmarsh True-3D world instantaneously in WASM.
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

            let hero_mesh = build_hero_knight_mesh();
            let hero_instances = showcase_hero_instances();

            let renderer = WebRenderer::from_canvas(
                canvas,
                w,
                h,
                &volume,
                &bricks,
                &sn_world,
                &hero_mesh,
                &hero_instances,
            )
            .await
            .map_err(|e| JsValue::from_str(&e))?;

            let cam = Camera3D::preset_bridge_waterfall();
            let capsule_pos = Vec3::new(113.5, 4.2, 140.0);

            Ok(FrostmarshApp {
                volume,
                renderer,
                cam,
                keys: HashSet::new(),
                pixel_art: false,
                capsule_mode: false,
                capsule_pos,
                capsule_vel: Vec3::ZERO,
                capsule_grounded: false,
                sim_time: 0.0,
                fps_smooth: 60.0,
                frame_ms_smooth: 16.6,
                volume_source,
                volume_ms,
                sn_build_ms,
                non_empty_bricks: bricks.non_empty_bricks,
                total_bricks: bricks.total_bricks,
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
                let fov_deg = (self.cam.fov_y.to_degrees() + delta_y * 0.025).clamp(32.0, 78.0);
                self.cam.fov_y = fov_deg.to_radians();
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
        }

        /// Toggle stylized `Pixel-2×SS` ↔ `Clean-3D` (`T`-key). Returns `true` when `Pixel-2×SS` is active.
        pub fn toggle_pixel_style(&mut self) -> bool {
            self.pixel_art = !self.pixel_art;
            self.pixel_art
        }

        /// Toggle 3D Capsule Collision Walk Mode (`C`-key). Returns `true` when Capsule Walk is active.
        pub fn toggle_capsule_mode(&mut self) -> bool {
            self.capsule_mode = !self.capsule_mode;
            if self.capsule_mode {
                // Place capsule smoothly below current camera eye (or onto bridge deck if flying high)
                let foot_y = (self.cam.pos.y - CAPSULE_EYE_HEIGHT)
                    .clamp(1.2, self.volume.depth as f32 - DEFAULT_CAPSULE_HEIGHT - 0.2);
                self.capsule_pos = Vec3::new(
                    self.cam.pos.x.clamp(2.0, self.volume.width as f32 - 2.0),
                    foot_y,
                    self.cam.pos.z.clamp(2.0, self.volume.height as f32 - 2.0),
                );
                self.capsule_vel = Vec3::ZERO;
                // Settle one quick step so we start cleanly above solid isosurface
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
                self.cam.pos = self.capsule_pos + Vec3::new(0.0, CAPSULE_EYE_HEIGHT, 0.0);
            }
            self.capsule_mode
        }

        /// Hot-swap to an external `FM3D` neural volume in browser.
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
            self.non_empty_bricks = bricks.non_empty_bricks;
            self.total_bricks = bricks.total_bricks;
            self.renderer.upload_volume(&volume, &bricks, &sn_world);
            self.volume = volume;
            self.volume_source = label;
            Ok(())
        }

        /// Hot-swap back to the built-in procedural isotropic `256×256×16` showcase world.
        pub fn load_procedural_showcase(&mut self) {
            let t0 = perf_now_ms();
            let volume = WorldVolume3D::generate_showcase();
            self.volume_ms = (perf_now_ms() - t0).max(0.0);
            let bricks = volume.build_brick_occupancy();
            let t_sn = perf_now_ms();
            let sn_world = SurfaceNetsWorld::build_all(&volume);
            self.sn_build_ms = (perf_now_ms() - t_sn).max(0.0);
            self.non_empty_bricks = bricks.non_empty_bricks;
            self.total_bricks = bricks.total_bricks;
            self.renderer.upload_volume(&volume, &bricks, &sn_world);
            self.volume = volume;
            self.volume_source = "Procedural Showcase (WASM)".to_string();
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
                self.cam.pos = self.capsule_pos + Vec3::new(0.0, CAPSULE_EYE_HEIGHT, 0.0);
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
            }

            self.renderer
                .render_frame(&self.cam, self.sim_time, self.pixel_art);
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
                    "\"walk_mode\":\"{}\",",
                    "\"grounded\":{},",
                    "\"cam_x\":{:.1},",
                    "\"cam_y\":{:.1},",
                    "\"cam_z\":{:.1},",
                    "\"volume_source\":\"{}\",",
                    "\"volume_ms\":{:.1},",
                    "\"sn_build_ms\":{:.1}",
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
                self.non_empty_bricks,
                self.total_bricks,
                style,
                walk,
                self.capsule_grounded,
                self.cam.pos.x,
                self.cam.pos.y,
                self.cam.pos.z,
                json_escape(&self.volume_source),
                self.volume_ms,
                self.sn_build_ms,
            )
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_app::FrostmarshApp;

#[cfg(test)]
mod tests {
    use crate::collision::{step_capsule_3d, DEFAULT_CAPSULE_HEIGHT, DEFAULT_CAPSULE_RADIUS};
    use crate::hero_mesh::{build_hero_knight_mesh, showcase_hero_instances};
    use crate::surface_nets_3d::SurfaceNetsWorld;
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
    fn test_fm3d_memory_parser() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FM3D");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        let n = 4 * 4 * 2 * 4;
        buf.extend(vec![128u8; n]);
        buf.extend(vec![200u8; n]);
        let vol = WorldVolume3D::from_fm3d_bytes(&buf).expect("parse FM3D");
        assert_eq!((vol.width, vol.height, vol.depth), (4, 4, 2));
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
