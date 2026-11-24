//! Frostmarsh Unified v2 Client ("Shattered Reach")
//!
//! Merges ALL 5 experimental tracks into a single scaled `wgpu 30` + `quinn` QUIC client:
//! - Approach 4: `O(N)` separable top-weighted smooth terrain, parametric CSG/SDF 3-class heroes
//!   & 3 capturable Shrine Obelisks, Hard-Z GPU-instanced SDF foliage surfels, 1 KB `flash()` palette,
//!   and 9-step Cool-Indigo `#261e48` / Warm-Cream `#fff4d6` 2× SS resolve.
//! - Tier 3 Hybrid: Depth-tested Soft Volumetric 3D Gaussian Splats (`volumetric_splat.wgsl`)
//!   strictly for leaping wildfire flames, charcoal smoke, scalding steam, spell trails & wisps.
//! - Native-Resolution GPU 2D Action-MMO HUD (`hud_pass.wgsl`): Match Score Banner, 3 Shrine Pips,
//!   Weather Badge, Live `256×256` Tactical Minimap, HP/Mana/Stamina/XP Bars, Status Pills,
//!   5-Slot Spell Hotbar with Cooldown Sweeps, Overhead HP Bars, Floating Popups, & GB10 Telemetry.
//!
//! Modes:
//! - Default (interactive): Native `winit 0.30` Metal / Vulkan playable client (`--server 100.116.218.113:4433`)
//! - `--headless`: Automated 1080p/4K capture + GPU FPS + 10 Hz `256×256` dirty-chunk benchmark

mod audio;
mod camera;
mod hud;
mod macro_planet_world;
mod mesh_gen;
mod net;
mod neural_style;
mod palette;
mod parametric;
mod planet_world;
mod renderer;
mod surfel_gen;
mod world_bridge;
mod z16_world;

use crate::audio::{SfxKind, SpatialAudioEngine};
use crate::camera::{
    CameraPreset, MacroPlanetCamMode, MacroPlanetCamera, MobaCamera, PlanetCamMode, PlanetCamera,
};
use crate::hud::{build_hud, HudConfig};
use crate::macro_planet_world::{
    MacroPlanetWorld, MACRO_MINIMAP_H, MACRO_MINIMAP_W,
};
use crate::mesh_gen::{StaticMeshBank, CLASS_BURROWER};
use crate::net::QuicClientHandle;
use crate::neural_style::NUM_Z16_STYLES;
use crate::planet_world::{PlanetWorld, PLANET_MINIMAP_H, PLANET_MINIMAP_W};
use crate::renderer::{z16_style_label, Approach4Renderer, Atmosphere, RenderStyle};
use crate::surfel_gen::SurfelProtoBank;
use crate::world_bridge::{assemble_live_frame, AimReticle, LiveApproach4World, SP_ICE, SP_ROCK, SP_WOOD};
use crate::z16_world::Z16World;
use clap::Parser;
use field_core::planet_macro::face_virtual_uv_f64;
use frostmarsh_proto::*;
use glam::{DVec3, Vec3};
use image::{ImageBuffer, Rgba};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

#[derive(Parser, Debug, Clone)]
#[command(name = "frostmarsh-unified-client", about = "Frostmarsh Unified v2 Scaled Client")]
struct Args {
    /// Live Frostmarsh QUIC Neural World Server address
    #[arg(long, default_value = "100.116.218.113:4433")]
    server: SocketAddr,
    /// Player display name
    #[arg(long, default_value = "Player")]
    name: String,
    /// Team (0 = West Frost, 1 = East Ember)
    #[arg(long, default_value_t = 0)]
    team: u8,
    /// Run headless capture & benchmark instead of opening an interactive window
    #[arg(long, default_value_t = false)]
    headless: bool,
    /// Initial resolve style: "pixel" or "clean"
    #[arg(long, default_value = "pixel")]
    style: String,
    /// Output directory for headless screenshots & reports
    #[arg(long, alias = "capture-dir", default_value = ".")]
    out_dir: PathBuf,
    /// Headless GPU benchmark iterations per mode
    #[arg(long, default_value_t = 40)]
    bench_iters: usize,
    /// Headless 10 Hz dirty-chunk observation window (seconds)
    #[arg(long, default_value_t = 4.0)]
    observe_secs: f64,
    /// Optional chat command automatically sent 1.5s into headless observation (e.g. "freeze north bridge")
    #[arg(long)]
    test_chat: Option<String>,
}

#[derive(Serialize)]
struct GpuBenchEntry {
    label: String,
    resolution: String,
    art_resolution: String,
    style: String,
    p50_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    fps_p50: f64,
}

#[derive(Serialize)]
struct UnifiedMetrics {
    server: String,
    gpu_adapter: String,
    gpu_backend: String,
    grid_w: usize,
    grid_h: usize,
    total_chunks: usize,
    parametric_build_ms: f64,
    sim_mode: String,
    backend: String,
    field_tick: u64,
    world_infer_ms: f32,
    eco_infer_ms: f32,
    proj_active_pct: f32,
    mass_error: i64,
    energy_error: i64,
    wire_kbps: f32,
    aoi_chunks_sent: u16,
    active_bots: u16,
    weather_kind: u8,
    dirty_update_p50_ms: f64,
    dirty_update_p99_ms: f64,
    full_sync_build_ms: f64,
    visible_chunks: usize,
    terrain_triangles: usize,
    mesh_instances: usize,
    surfel_draws: usize,
    volumetric_splats: usize,
    hud_vertices: usize,
    gpu_benchmarks: Vec<GpuBenchEntry>,
}

fn save_rgba_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w, h, rgba.to_vec()).ok_or("invalid image buffer")?;
    img.save(path)?;
    Ok(())
}

fn compose_2x2_grid(panels: [&[u8]; 4], pw: u32, ph: u32) -> Vec<u8> {
    let out_w = pw * 2;
    let out_h = ph * 2;
    let mut out = vec![0u8; (out_w * out_h * 4) as usize];
    for (pi, src) in panels.iter().enumerate() {
        let ox = (pi as u32 % 2) * pw;
        let oy = (pi as u32 / 2) * ph;
        for y in 0..ph {
            let dst_off = (((oy + y) * out_w + ox) * 4) as usize;
            let src_off = (y * pw * 4) as usize;
            let row = (pw * 4) as usize;
            out[dst_off..dst_off + row].copy_from_slice(&src[src_off..src_off + row]);
        }
    }
    out
}

fn style_from_name(name: &str, out_h: u32) -> RenderStyle {
    if name == "clean" {
        RenderStyle::clean_3d()
    } else if out_h >= 1800 {
        RenderStyle::pixel_art(4)
    } else {
        RenderStyle::pixel_art(2)
    }
}

// =========================================================================
// HEADLESS CAPTURE & BENCHMARK MODE (1080p & 4K on GB10 / Desktop GPU)
// =========================================================================

fn run_headless(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let t_param = Instant::now();
    let meshes = StaticMeshBank::build(1);
    let surfels = SurfelProtoBank::build();
    let parametric_build_ms = t_param.elapsed().as_secs_f64() * 1000.0;

    let mut renderer = rt.block_on(Approach4Renderer::new_headless(&meshes, &surfels))?;
    println!(
        "[Unified] Headless GPU: {} ({}) | Parametric build {:.2}ms",
        renderer.adapter_name, renderer.adapter_backend, parametric_build_ms
    );

    let quic = rt.block_on(QuicClientHandle::connect(args.server, &args.name, args.team))?;
    let (rc, tot, has_ent) = rt.block_on(quic.wait_for_sync(0.95, Duration::from_secs(10)));
    println!("[Unified] Synced {}/{} chunks/bricks (entities={})", rc, tot, has_ent);

    let (gw, gh, gd, z16_active, macro_planet_mode, planet_mode, planet_r_core, planet_shards_per_axis, planet_seed) = {
        let st = quic.state.lock().unwrap();
        let mmode = st.macro_planet_mode
            || st.received_macro_shard_brick_count() > 0
            || st.received_macro_climate_face_count() > 0
            || st.entities.as_ref().is_some_and(|e| e.telemetry.macro_planet);
        let pmode = !mmode
            && (st.planet_mode
                || st.received_planet_brick_count() > 0
                || st.entities.as_ref().is_some_and(|e| e.telemetry.planet_mode));
        let r_core = st
            .entities
            .as_ref()
            .map(|e| {
                if e.telemetry.planet_r_core > 1.0 {
                    e.telemetry.planet_r_core
                } else if mmode {
                    1280.0
                } else {
                    st.planet_r_core
                }
            })
            .unwrap_or(if mmode { 1280.0 } else { st.planet_r_core });
        let spa = st
            .entities
            .as_ref()
            .map(|e| if e.telemetry.planet_shards_per_axis > 0 { e.telemetry.planet_shards_per_axis } else { st.planet_shards_per_axis })
            .unwrap_or(st.planet_shards_per_axis);
        let pseed = st
            .entities
            .as_ref()
            .map(|e| if e.telemetry.planet_seed != 0 { e.telemetry.planet_seed } else { st.planet_seed })
            .unwrap_or(st.planet_seed);
        (
            st.width,
            st.height,
            st.depth,
            !mmode && !pmode && (st.depth >= 16 || st.has_z16_bricks),
            mmode,
            pmode,
            r_core,
            spa,
            pseed,
        )
    };

    // Wave 34B: Dedicated Live Dual-Scale Locally-Flat R=1,304m Macro Planet Headless Capture & Metal Benchmark
    if macro_planet_mode {
        let mut macro_world = MacroPlanetWorld::new(
            (planet_r_core as f64).max(100.0),
            planet_shards_per_axis.max(1),
            gd.max(16),
            planet_seed,
        );
        let wait_start = Instant::now();
        let mut last_sync_req = Instant::now();
        loop {
            let (clim_u, shard_u, has_e, r_c, spa, sd, cur_d) = {
                let mut st = quic.state.lock().unwrap();
                let cu = st.take_macro_climate_updates();
                let su = st.take_macro_shard_brick_updates();
                let r_c = st
                    .entities
                    .as_ref()
                    .map(|e| if e.telemetry.planet_r_core > 1.0 { e.telemetry.planet_r_core as f64 } else { 1280.0 })
                    .unwrap_or(1280.0);
                let spa = st
                    .entities
                    .as_ref()
                    .map(|e| if e.telemetry.planet_shards_per_axis > 0 { e.telemetry.planet_shards_per_axis } else { st.planet_shards_per_axis })
                    .unwrap_or(st.planet_shards_per_axis);
                let sd = st
                    .entities
                    .as_ref()
                    .map(|e| if e.telemetry.planet_seed != 0 { e.telemetry.planet_seed } else { st.planet_seed })
                    .unwrap_or(st.planet_seed);
                (cu, su, st.entities.is_some(), r_c, spa, sd, st.depth.max(16))
            };
            macro_world.configure_if_needed(r_c, spa, cur_d, sd);
            macro_world.apply_climate_updates(clim_u);
            macro_world.apply_shard_updates(shard_u);
            if macro_world.received_brick_count() >= 32
                && macro_world.received_climate_face_count() >= 6
                && has_e
            {
                break;
            }
            if wait_start.elapsed() >= Duration::from_secs(12) {
                break;
            }
            if last_sync_req.elapsed() >= Duration::from_millis(1200) {
                let _ = rt.block_on(QuicClientHandle::send_reliable_on(
                    &quic.conn,
                    &ClientReliable::RequestFullSync,
                ));
                last_sync_req = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(25));
        }

        let entities_init = { quic.state.lock().unwrap().entities.clone() };
        let sun_init = entities_init
            .as_ref()
            .map(|e| {
                let s = DVec3::new(e.telemetry.sun_dir[0] as f64, e.telemetry.sun_dir[1] as f64, e.telemetry.sun_dir[2] as f64);
                if s.length_squared() > 0.01 { s.normalize() } else { DVec3::new(-0.72, 0.36, 0.59).normalize() }
            })
            .unwrap_or_else(|| DVec3::new(-0.72, 0.36, 0.59).normalize());
        let (focus_pos0, _, _) = macro_world.find_hero_or_best_cluster_f64(entities_init.as_ref(), sun_init.to_array());
        macro_world.ensure_floating_origin(focus_pos0);

        let t_build = Instant::now();
        let init_rebuilt = macro_world.rebuild_dirty(12);
        let full_sync_build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[Unified-MacroPlanet] Initial Dual-Scale Clipmap: {} hydrated shards ({} bricks, {}/6 climate faces), {} tris in {:.2}ms",
            init_rebuilt,
            macro_world.received_brick_count(),
            macro_world.received_climate_face_count(),
            macro_world.total_triangles(),
            full_sync_build_ms
        );

        let mut dirty_samples = Vec::new();
        let t_obs = Instant::now();
        while t_obs.elapsed().as_secs_f64() < args.observe_secs {
            let (cu, su) = {
                let mut st = quic.state.lock().unwrap();
                (st.take_macro_climate_updates(), st.take_macro_shard_brick_updates())
            };
            if !cu.is_empty() || !su.is_empty() {
                let t0 = Instant::now();
                macro_world.apply_climate_updates(cu);
                macro_world.apply_shard_updates(su);
                let cnt = macro_world.rebuild_dirty(6);
                if cnt > 0 || !macro_world.climate_dirty_faces.iter().all(|&d| !d) {
                    dirty_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
            }
            std::thread::sleep(Duration::from_millis(12));
        }
        let _ = macro_world.rebuild_dirty(12);
        dirty_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let dirty_p50 = dirty_samples.get(dirty_samples.len() / 2).copied().unwrap_or(0.0);
        let dirty_p99 = dirty_samples
            .get(((dirty_samples.len() as f64) * 0.99).floor().min(dirty_samples.len().saturating_sub(1) as f64) as usize)
            .copied()
            .unwrap_or(dirty_p50);

        let entities = { quic.state.lock().unwrap().entities.clone() };
        let sun_d = entities
            .as_ref()
            .map(|e| {
                let s = DVec3::new(e.telemetry.sun_dir[0] as f64, e.telemetry.sun_dir[1] as f64, e.telemetry.sun_dir[2] as f64);
                if s.length_squared() > 0.01 { s.normalize() } else { DVec3::new(-0.72, 0.36, 0.59).normalize() }
            })
            .unwrap_or_else(|| DVec3::new(-0.72, 0.36, 0.59).normalize());
        let sun_f = [sun_d.x as f32, sun_d.y as f32, sun_d.z as f32];
        let local_clim = entities
            .as_ref()
            .map(|e| [e.telemetry.local_wind_u, e.telemetry.local_wind_v, e.telemetry.local_cloud_cover])
            .unwrap_or([0.0, 0.0, 0.35]);

        let (focus_pos, focus_fwd, isl_dir) =
            macro_world.find_hero_or_best_cluster_f64(entities.as_ref(), sun_d.to_array());
        if macro_world.ensure_floating_origin(focus_pos) {
            let _ = macro_world.rebuild_dirty(12);
        }

        let sim_time = 2.4f32;
        macro_world.rebuild_dynamic_entities(entities.as_ref(), sim_time);
        renderer.upload_macro_planet_world(&mut macro_world);
        renderer.upload_minimap(
            MACRO_MINIMAP_W as u32,
            MACRO_MINIMAP_H as u32,
            &macro_world.minimap_rgba,
        );

        let world_dummy = LiveApproach4World::new(256, 256, 4);
        let r_surf = macro_world.r_surf();
        let cam_ground = MacroPlanetCamera::ground_3rd_person_shot(
            DVec3::from_array(focus_pos),
            DVec3::from_array(focus_fwd),
            DVec3::from_array(isl_dir),
        );
        let cam_flight = MacroPlanetCamera::skywing_flight_shot(
            DVec3::from_array(focus_pos),
            DVec3::from_array(focus_fwd),
            DVec3::from_array(isl_dir),
            r_surf,
        );
        let cam_strato = MacroPlanetCamera::stratosphere_shot(
            DVec3::from_array(focus_pos),
            DVec3::from_array(focus_fwd),
            r_surf,
        );
        let cam_orbital = MacroPlanetCamera::orbital_globe_shot(sun_d, r_surf);

        let mut gpu_benchmarks = Vec::new();
        let mut fps_1080 = 300.0f32;
        for (label, bw, bh, bcam) in [
            ("1080p Macro Ground 1m + HUD", 1920u32, 1080u32, &cam_ground),
            ("4K Macro Ground 1m + HUD", 3840u32, 2160u32, &cam_ground),
            ("1080p Macro Orbital + Climate", 1920u32, 1080u32, &cam_orbital),
            ("4K Macro Orbital + Climate", 3840u32, 2160u32, &cam_orbital),
        ] {
            let mm = bcam.compute_matrices(bw as f32, bh as f32);
            let hm = mm.to_hud_camera_matrices();
            let hb = build_hud(
                bw as f32,
                bh as f32,
                &world_dummy,
                None,
                entities.as_ref(),
                &hm,
                &HudConfig {
                    show_hud: true,
                    show_telemetry: true,
                    show_brain_inspector: true,
                    brain_selected_idx: None,
                    selected_element: 0,
                    focus_radius: 4.0,
                    style_label: "MACRO-3D",
                    z16_active: false,
                    z16_style_label: "MACRO",
                    fps: fps_1080,
                    rtt_ms: 1.2,
                    chat_open: false,
                    chat_input: "",
                    chat_shout: true,
                    show_scoreboard: false,
                    audio_muted: true,
                    audio_available: false,
                    cave_enclosure: 0.0,
                    show_squad_links: true,
                    camera_preset_label: bcam.mode.label(),
                    macro_eye_f64: Some(mm.eye_f64),
                },
            );
            let (p50, p99, mean) = renderer.benchmark_macro_planet_gpu(
                bw,
                bh,
                &mm,
                sun_f,
                bcam.exposure,
                local_clim,
                &macro_world,
                Some(&hb),
                6,
                args.bench_iters,
            );
            let fps = 1000.0 / p50.max(0.001);
            if bw == 1920 && bcam.mode == MacroPlanetCamMode::Ground3rdPerson {
                fps_1080 = fps as f32;
            }
            println!(
                "  [MacroPlanet Bench] {:32} | p50={:.3}ms p99={:.3}ms ({:.1} FPS)",
                label, p50, p99, fps
            );
            gpu_benchmarks.push(GpuBenchEntry {
                label: label.to_string(),
                resolution: format!("{}x{}", bw, bh),
                art_resolution: format!("{}x{}", bw, bh),
                style: "MACRO-PLANET-METAL".to_string(),
                p50_ms: (p50 * 1000.0).round() / 1000.0,
                p99_ms: (p99 * 1000.0).round() / 1000.0,
                mean_ms: (mean * 1000.0).round() / 1000.0,
                fps_p50: (fps * 10.0).round() / 10.0,
            });
        }

        std::fs::create_dir_all(&args.out_dir)?;

        // Shot 1: `unified_live_1080p_hud.png` — Ground 3rd-Person Hero Shot (with HUD)
        let mats_g_1080 = cam_ground.compute_matrices(1920.0, 1080.0);
        let hm_g = mats_g_1080.to_hud_camera_matrices();
        let hud_main = build_hud(
            1920.0,
            1080.0,
            &world_dummy,
            None,
            entities.as_ref(),
            &hm_g,
            &HudConfig {
                show_hud: true,
                show_telemetry: true,
                show_brain_inspector: true,
                brain_selected_idx: None,
                selected_element: 0,
                focus_radius: 4.0,
                style_label: "MACRO-3D",
                z16_active: false,
                z16_style_label: "MACRO",
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
                cave_enclosure: 0.0,
                show_squad_links: true,
                camera_preset_label: MacroPlanetCamMode::Ground3rdPerson.label(),
                macro_eye_f64: Some(mats_g_1080.eye_f64),
            },
        );
        let (rgba_g, _) = renderer.render_macro_planet_to_rgba(
            1920,
            1080,
            &mats_g_1080,
            sun_f,
            cam_ground.exposure,
            sim_time,
            local_clim,
            &macro_world,
            Some(&hud_main),
        );
        save_rgba_png(&args.out_dir.join("unified_live_1080p_hud.png"), 1920, 1080, &rgba_g)?;

        // Shot 2: `unified_closeup_action_1080p.png` — Skywing Flight Action Shot (~95m altitude)
        let mats_fl_1080 = cam_flight.compute_matrices(1920.0, 1080.0);
        let hm_fl = mats_fl_1080.to_hud_camera_matrices();
        let hud_fl = build_hud(
            1920.0,
            1080.0,
            &world_dummy,
            None,
            entities.as_ref(),
            &hm_fl,
            &HudConfig {
                show_hud: true,
                show_telemetry: true,
                show_brain_inspector: true,
                brain_selected_idx: None,
                selected_element: 0,
                focus_radius: 4.0,
                style_label: "MACRO-3D",
                z16_active: false,
                z16_style_label: "MACRO",
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
                cave_enclosure: 0.0,
                show_squad_links: true,
                camera_preset_label: MacroPlanetCamMode::SkywingFlight.label(),
                macro_eye_f64: Some(mats_fl_1080.eye_f64),
            },
        );
        let (rgba_fl, _) = renderer.render_macro_planet_to_rgba(
            1920,
            1080,
            &mats_fl_1080,
            sun_f,
            cam_flight.exposure,
            sim_time,
            local_clim,
            &macro_world,
            Some(&hud_fl),
        );
        save_rgba_png(&args.out_dir.join("unified_closeup_action_1080p.png"), 1920, 1080, &rgba_fl)?;

        // Shot 3: `unified_orbital_1080p.png` — Full Orbital Globe (3,250m) with live Coriolis storm clouds!
        let mats_orb_1080 = cam_orbital.compute_matrices(1920.0, 1080.0);
        let hm_orb = mats_orb_1080.to_hud_camera_matrices();
        let hud_orb = build_hud(
            1920.0,
            1080.0,
            &world_dummy,
            None,
            entities.as_ref(),
            &hm_orb,
            &HudConfig {
                show_hud: true,
                show_telemetry: true,
                show_brain_inspector: false,
                brain_selected_idx: None,
                selected_element: 0,
                focus_radius: 4.0,
                style_label: "MACRO-3D",
                z16_active: false,
                z16_style_label: "MACRO",
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
                cave_enclosure: 0.0,
                show_squad_links: true,
                camera_preset_label: MacroPlanetCamMode::OrbitalGlobe.label(),
                macro_eye_f64: Some(mats_orb_1080.eye_f64),
            },
        );
        let (rgba_orb, _) = renderer.render_macro_planet_to_rgba(
            1920,
            1080,
            &mats_orb_1080,
            sun_f,
            cam_orbital.exposure,
            sim_time,
            local_clim,
            &macro_world,
            Some(&hud_orb),
        );
        save_rgba_png(&args.out_dir.join("unified_orbital_1080p.png"), 1920, 1080, &rgba_orb)?;

        // Shot 4: `unified_orbit_4view.png` — 2×2 Live Dual-Scale Transition Showcase
        //   (Ground 1m Flat + HUD | Skywing Flight | Stratosphere Limb | Full Orbital Globe w/ Coriolis Clouds)
        let q_cams = [&cam_ground, &cam_flight, &cam_strato, &cam_orbital];
        let mut panels = Vec::with_capacity(4);
        for (qi, qcam) in q_cams.iter().enumerate() {
            let qm = qcam.compute_matrices(960.0, 540.0);
            let qhud = if qi == 0 {
                let hm = qm.to_hud_camera_matrices();
                Some(build_hud(
                    960.0,
                    540.0,
                    &world_dummy,
                    None,
                    entities.as_ref(),
                    &hm,
                    &HudConfig {
                        show_hud: true,
                        show_telemetry: true,
                        show_brain_inspector: false,
                        brain_selected_idx: None,
                        selected_element: 0,
                        focus_radius: 4.0,
                        style_label: "MACRO-3D",
                        z16_active: false,
                        z16_style_label: "MACRO",
                        fps: fps_1080,
                        rtt_ms: 1.2,
                        chat_open: false,
                        chat_input: "",
                        chat_shout: true,
                        show_scoreboard: false,
                        audio_muted: true,
                        audio_available: false,
                        cave_enclosure: 0.0,
                        show_squad_links: true,
                        camera_preset_label: qcam.mode.label(),
                        macro_eye_f64: Some(qm.eye_f64),
                    },
                ))
            } else {
                None
            };
            let (px, _) = renderer.render_macro_planet_to_rgba(
                960,
                540,
                &qm,
                sun_f,
                qcam.exposure,
                sim_time,
                local_clim,
                &macro_world,
                qhud.as_ref(),
            );
            panels.push(px);
        }
        let grid = compose_2x2_grid([&panels[0], &panels[1], &panels[2], &panels[3]], 960, 540);
        save_rgba_png(&args.out_dir.join("unified_orbit_4view.png"), 1920, 1080, &grid)?;

        let tel = entities.as_ref().map(|e| e.telemetry.clone());
        let metrics = UnifiedMetrics {
            server: args.server.to_string(),
            gpu_adapter: renderer.adapter_name.clone(),
            gpu_backend: renderer.adapter_backend.clone(),
            grid_w: macro_world.face_size_virtual(),
            grid_h: macro_world.face_size_virtual(),
            total_chunks: macro_world.received_brick_count(),
            parametric_build_ms: (parametric_build_ms * 100.0).round() / 100.0,
            sim_mode: tel.as_ref().map(|t| t.mode.clone()).unwrap_or_else(|| "macro-planet".to_string()),
            backend: tel.as_ref().map(|t| t.backend.clone()).unwrap_or_default(),
            field_tick: tel.as_ref().map(|t| t.field_tick).unwrap_or(0),
            world_infer_ms: tel.as_ref().map(|t| t.infer_ms_last).unwrap_or(0.0),
            eco_infer_ms: tel.as_ref().map(|t| t.eco_infer_ms).unwrap_or(0.0),
            proj_active_pct: tel.as_ref().map(|t| t.proj_active_pct).unwrap_or(0.0),
            mass_error: tel.as_ref().map(|t| t.mass_error).unwrap_or(0),
            energy_error: tel.as_ref().map(|t| t.energy_error).unwrap_or(0),
            wire_kbps: tel.as_ref().map(|t| t.wire_kbps).unwrap_or(0.0),
            aoi_chunks_sent: macro_world.received_brick_count() as u16,
            active_bots: tel.as_ref().map(|t| t.active_bots).unwrap_or(0),
            weather_kind: tel.as_ref().map(|t| t.weather_kind).unwrap_or(0),
            dirty_update_p50_ms: (dirty_p50 * 1000.0).round() / 1000.0,
            dirty_update_p99_ms: (dirty_p99 * 1000.0).round() / 1000.0,
            full_sync_build_ms: (full_sync_build_ms * 1000.0).round() / 1000.0,
            visible_chunks: macro_world.hydrated_shard_count(),
            terrain_triangles: macro_world.total_triangles(),
            mesh_instances: entities.as_ref().map(|e| e.players.len() + e.shrines.len()).unwrap_or(0),
            surfel_draws: 0,
            volumetric_splats: macro_world.near_gas_indices.len() / 3,
            hud_vertices: hud_main.vertices.len(),
            gpu_benchmarks,
        };
        let json = serde_json::to_string_pretty(&metrics)?;
        std::fs::write(args.out_dir.join("unified_metrics.json"), &json)?;
        println!("[Unified-MacroPlanet] Saved unified_live_1080p_hud.png, unified_closeup_action_1080p.png, unified_orbital_1080p.png, unified_orbit_4view.png, unified_metrics.json");
        return Ok(());
    }

    // Wave 31B: Dedicated Live 6-Face Spherical Planetary Headless Capture & Metal Benchmark
    if planet_mode {
        let mut planet_world = PlanetWorld::new(gw.max(16), gd.max(16), planet_r_core.max(16.0));
        let wait_start = Instant::now();
        let mut last_sync_req = Instant::now();
        loop {
            let (updates, has_e, face_n, depth_d, r_c) = {
                let mut st = quic.state.lock().unwrap();
                let u = st.take_planet_brick_updates();
                let rc = st
                    .entities
                    .as_ref()
                    .map(|e| if e.telemetry.planet_r_core > 1.0 { e.telemetry.planet_r_core } else { st.planet_r_core })
                    .unwrap_or(st.planet_r_core);
                (u, st.entities.is_some(), st.width, st.height.max(st.depth), rc)
            };
            let _ = depth_d;
            let cur_depth = quic.state.lock().unwrap().depth.max(16);
            planet_world.resize_if_needed(face_n.max(16), cur_depth, r_c.max(16.0));
            planet_world.apply_updates(updates);
            if planet_world.received_brick_count() >= 48 && has_e {
                break;
            }
            if wait_start.elapsed() >= Duration::from_secs(12) {
                break;
            }
            if last_sync_req.elapsed() >= Duration::from_millis(1200) {
                let _ = rt.block_on(QuicClientHandle::send_reliable_on(
                    &quic.conn,
                    &ClientReliable::RequestFullSync,
                ));
                last_sync_req = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(25));
        }

        let t_build = Instant::now();
        let init_rebuilt = planet_world.rebuild_dirty_faces(6);
        let full_sync_build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[Unified-Planet] Initial 6-Face Spherical Build: {}/6 faces, {} bricks, {} tris in {:.2}ms",
            init_rebuilt,
            planet_world.received_brick_count(),
            planet_world.total_triangles(),
            full_sync_build_ms
        );

        // Observe live 10 Hz planetary brick + delta streaming
        let mut dirty_samples = Vec::new();
        let t_obs = Instant::now();
        while t_obs.elapsed().as_secs_f64() < args.observe_secs {
            let updates = { quic.state.lock().unwrap().take_planet_brick_updates() };
            if !updates.is_empty() {
                let t0 = Instant::now();
                planet_world.apply_updates(updates);
                let cnt = planet_world.rebuild_dirty_faces(6);
                if cnt > 0 {
                    dirty_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
            }
            std::thread::sleep(Duration::from_millis(12));
        }
        let _ = planet_world.rebuild_dirty_faces(6);
        dirty_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let dirty_p50 = dirty_samples.get(dirty_samples.len() / 2).copied().unwrap_or(0.0);
        let dirty_p99 = dirty_samples
            .get(((dirty_samples.len() as f64) * 0.99).floor().min(dirty_samples.len().saturating_sub(1) as f64) as usize)
            .copied()
            .unwrap_or(dirty_p50);

        let entities = { quic.state.lock().unwrap().entities.clone() };
        let sun_dir = entities
            .as_ref()
            .map(|e| {
                let s = Vec3::from_array(e.telemetry.sun_dir);
                if s.length_squared() > 0.01 {
                    s.normalize()
                } else {
                    Vec3::new(-0.78, 0.28, 0.56).normalize()
                }
            })
            .unwrap_or_else(|| Vec3::new(-0.78, 0.28, 0.56).normalize());

        let sim_time = 2.4f32;
        planet_world.rebuild_dynamic_entities(entities.as_ref(), sim_time);
        renderer.upload_planet_world(&mut planet_world);
        renderer.upload_minimap(
            PLANET_MINIMAP_W as u32,
            PLANET_MINIMAP_H as u32,
            &planet_world.minimap_rgba,
        );

        let world_dummy = LiveApproach4World::new(gw.max(16), gh.max(16), 4);
        let r_outer = planet_world.r_outer();
        let cam_orbital = PlanetCamera::orbital_terminator_shot(sun_dir, r_outer);
        let mats_orb_1080 = cam_orbital.compute_matrices(1920.0, 1080.0);
        let hud_orb_1080_cam = mats_orb_1080.to_camera_matrices();

        // Benchmark 1080p & 4K Live Curved Planet Metal FPS
        let mut gpu_benchmarks = Vec::new();
        let mut fps_1080 = 300.0f32;
        for (label, bw, bh) in [
            ("1080p Curved Planet + 60 Bots + HUD", 1920u32, 1080u32),
            ("4K Curved Planet + 60 Bots + HUD", 3840u32, 2160u32),
        ] {
            let pm = cam_orbital.compute_matrices(bw as f32, bh as f32);
            let hm = pm.to_camera_matrices();
            let hb = build_hud(
                bw as f32,
                bh as f32,
                &world_dummy,
                None,
                entities.as_ref(),
                &hm,
                &HudConfig {
                    show_hud: true,
                    show_telemetry: true,
                    show_brain_inspector: true,
                    brain_selected_idx: None,
                    selected_element: 0,
                    focus_radius: 4.0,
                    style_label: "PLANET-3D",
                    z16_active: false,
                    z16_style_label: "PLANET",
                    fps: fps_1080,
                    rtt_ms: 1.2,
                    chat_open: false,
                    chat_input: "",
                    chat_shout: true,
                    show_scoreboard: false,
                    audio_muted: true,
                    audio_available: false,
                    cave_enclosure: 0.0,
                    show_squad_links: true,
                    camera_preset_label: PlanetCamMode::OrbitalGlobe.label(),
                    macro_eye_f64: None,
                },
            );
            let (p50, p99, mean) = renderer.benchmark_planet_gpu(
                bw,
                bh,
                &pm,
                sun_dir,
                cam_orbital.exposure,
                &planet_world,
                Some(&hb),
                6,
                args.bench_iters,
            );
            let fps = 1000.0 / p50.max(0.001);
            if bw == 1920 {
                fps_1080 = fps as f32;
            }
            println!(
                "  [Planet Bench] {:36} | p50={:.3}ms p99={:.3}ms ({:.1} FPS)",
                label, p50, p99, fps
            );
            gpu_benchmarks.push(GpuBenchEntry {
                label: label.to_string(),
                resolution: format!("{}x{}", bw, bh),
                art_resolution: format!("{}x{}", bw, bh),
                style: "PLANET-METAL".to_string(),
                p50_ms: (p50 * 1000.0).round() / 1000.0,
                p99_ms: (p99 * 1000.0).round() / 1000.0,
                mean_ms: (mean * 1000.0).round() / 1000.0,
                fps_p50: (fps * 10.0).round() / 10.0,
            });
        }

        std::fs::create_dir_all(&args.out_dir)?;

        // Shot 1: `unified_live_1080p_hud.png` — Full-Globe Orbital Terminator Hero Shot (with HUD)
        let hud_main = build_hud(
            1920.0,
            1080.0,
            &world_dummy,
            None,
            entities.as_ref(),
            &hud_orb_1080_cam,
            &HudConfig {
                show_hud: true,
                show_telemetry: true,
                show_brain_inspector: true,
                brain_selected_idx: None,
                selected_element: 0,
                focus_radius: 4.0,
                style_label: "PLANET-3D",
                z16_active: false,
                z16_style_label: "PLANET",
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
                cave_enclosure: 0.0,
                show_squad_links: true,
                camera_preset_label: PlanetCamMode::OrbitalGlobe.label(),
                macro_eye_f64: None,
            },
        );
        let (rgba_orb, _) = renderer.render_planet_to_rgba(
            1920,
            1080,
            &mats_orb_1080,
            sun_dir,
            cam_orbital.exposure,
            sim_time,
            &planet_world,
            Some(&hud_main),
        );
        save_rgba_png(&args.out_dir.join("unified_live_1080p_hud.png"), 1920, 1080, &rgba_orb)?;

        // Shot 2: `unified_closeup_action_1080p.png` — Curved Surface Horizon Action View
        let (focus_pos, focus_fwd) = planet_world.find_best_action_cluster(entities.as_ref(), sun_dir);
        let cam_surface = PlanetCamera::surface_horizon_battle_shot(focus_pos, focus_fwd);
        let mats_surf_1080 = cam_surface.compute_matrices(1920.0, 1080.0);
        let hud_surf_cam = mats_surf_1080.to_camera_matrices();
        let hud_surf = build_hud(
            1920.0,
            1080.0,
            &world_dummy,
            None,
            entities.as_ref(),
            &hud_surf_cam,
            &HudConfig {
                show_hud: true,
                show_telemetry: true,
                show_brain_inspector: true,
                brain_selected_idx: None,
                selected_element: 0,
                focus_radius: 4.0,
                style_label: "PLANET-3D",
                z16_active: false,
                z16_style_label: "PLANET",
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
                cave_enclosure: 0.0,
                show_squad_links: true,
                camera_preset_label: PlanetCamMode::SurfaceHorizon.label(),
                macro_eye_f64: None,
            },
        );
        let (rgba_surf, _) = renderer.render_planet_to_rgba(
            1920,
            1080,
            &mats_surf_1080,
            sun_dir,
            cam_surface.exposure,
            sim_time,
            &planet_world,
            Some(&hud_surf),
        );
        save_rgba_png(&args.out_dir.join("unified_closeup_action_1080p.png"), 1920, 1080, &rgba_surf)?;

        // Shot 3: `unified_orbit_4view.png` — 2×2 Live Planetary Battle Showcase Grid
        let cam_polar = PlanetCamera::polar_sky_ring_shot(r_outer);
        let cam_night = PlanetCamera::night_volcanic_rift_shot(sun_dir, r_outer);
        let q_cams = [&cam_orbital, &cam_surface, &cam_polar, &cam_night];
        let mut panels = Vec::with_capacity(4);
        for (qi, qcam) in q_cams.iter().enumerate() {
            let qm = qcam.compute_matrices(960.0, 540.0);
            let qhud = if qi == 0 {
                let hm = qm.to_camera_matrices();
                Some(build_hud(
                    960.0,
                    540.0,
                    &world_dummy,
                    None,
                    entities.as_ref(),
                    &hm,
                    &HudConfig {
                        show_hud: true,
                        show_telemetry: true,
                        show_brain_inspector: false,
                        brain_selected_idx: None,
                        selected_element: 0,
                        focus_radius: 4.0,
                        style_label: "PLANET-3D",
                        z16_active: false,
                        z16_style_label: "PLANET",
                        fps: fps_1080,
                        rtt_ms: 1.2,
                        chat_open: false,
                        chat_input: "",
                        chat_shout: true,
                        show_scoreboard: false,
                        audio_muted: true,
                        audio_available: false,
                        cave_enclosure: 0.0,
                        show_squad_links: true,
                        camera_preset_label: qcam.mode.label(),
                        macro_eye_f64: None,
                    },
                ))
            } else {
                None
            };
            let (px, _) = renderer.render_planet_to_rgba(
                960,
                540,
                &qm,
                sun_dir,
                qcam.exposure,
                sim_time,
                &planet_world,
                qhud.as_ref(),
            );
            panels.push(px);
        }
        let grid = compose_2x2_grid([&panels[0], &panels[1], &panels[2], &panels[3]], 960, 540);
        save_rgba_png(&args.out_dir.join("unified_orbit_4view.png"), 1920, 1080, &grid)?;

        let tel = entities.as_ref().map(|e| e.telemetry.clone());
        let metrics = UnifiedMetrics {
            server: args.server.to_string(),
            gpu_adapter: renderer.adapter_name.clone(),
            gpu_backend: renderer.adapter_backend.clone(),
            grid_w: planet_world.face_size,
            grid_h: planet_world.face_size,
            total_chunks: planet_world.received_brick_count(),
            parametric_build_ms: (parametric_build_ms * 100.0).round() / 100.0,
            sim_mode: tel.as_ref().map(|t| t.mode.clone()).unwrap_or_else(|| "planet".to_string()),
            backend: tel.as_ref().map(|t| t.backend.clone()).unwrap_or_default(),
            field_tick: tel.as_ref().map(|t| t.field_tick).unwrap_or(0),
            world_infer_ms: tel.as_ref().map(|t| t.infer_ms_last).unwrap_or(0.0),
            eco_infer_ms: tel.as_ref().map(|t| t.eco_infer_ms).unwrap_or(0.0),
            proj_active_pct: tel.as_ref().map(|t| t.proj_active_pct).unwrap_or(0.0),
            mass_error: tel.as_ref().map(|t| t.mass_error).unwrap_or(0),
            energy_error: tel.as_ref().map(|t| t.energy_error).unwrap_or(0),
            wire_kbps: tel.as_ref().map(|t| t.wire_kbps).unwrap_or(0.0),
            aoi_chunks_sent: planet_world.received_brick_count() as u16,
            active_bots: tel.as_ref().map(|t| t.active_bots).unwrap_or(0),
            weather_kind: tel.as_ref().map(|t| t.weather_kind).unwrap_or(0),
            dirty_update_p50_ms: (dirty_p50 * 1000.0).round() / 1000.0,
            dirty_update_p99_ms: (dirty_p99 * 1000.0).round() / 1000.0,
            full_sync_build_ms: (full_sync_build_ms * 1000.0).round() / 1000.0,
            visible_chunks: planet_world.populated_face_count(),
            terrain_triangles: planet_world.total_triangles(),
            mesh_instances: entities.as_ref().map(|e| e.players.len() + e.shrines.len()).unwrap_or(0),
            surfel_draws: 0,
            volumetric_splats: planet_world.combined_gas_indices.len() / 3,
            hud_vertices: hud_main.vertices.len(),
            gpu_benchmarks,
        };
        let json = serde_json::to_string_pretty(&metrics)?;
        std::fs::write(args.out_dir.join("unified_metrics.json"), &json)?;
        println!("[Unified-Planet] Saved unified_live_1080p_hud.png, unified_closeup_action_1080p.png, unified_orbit_4view.png, unified_metrics.json");
        return Ok(());
    }
    let mut world = LiveApproach4World::new(gw, gh, gd);
    let mut z16_world = if z16_active { Some(Z16World::new(gw, gh, gd)) } else { None };

    // Full initial 3D volumetric build (either true 3D Z=16..64 Surface-Nets OR L4 layered A+)
    let full_sync_build_ms = if let Some(ref mut z16w) = z16_world {
        let mut st = quic.state.lock().unwrap();
        let (rebuilt, ms) = if st.depth > 32 {
            let dirty_subs = st.take_dirty_z16_subbricks();
            z16w.apply_dirty_subbricks(&st.z16_density_mat, &st.z16_albedo_thermal, &dirty_subs)
        } else {
            let dirty_z16 = st.take_dirty_z16_bricks();
            z16w.apply_dirty_bricks(&st.z16_density_mat, &st.z16_albedo_thermal, &dirty_z16)
        };
        println!(
            "[Unified] Z={} Initial Surface-Nets: {} active bricks/sub-bricks, {} tris in {:.2}ms (bytes_rx={})",
            st.depth,
            rebuilt,
            z16w.total_triangles(),
            ms,
            st.bytes_rx
        );
        ms
    } else {
        let dirty_init = {
            let mut st = quic.state.lock().unwrap();
            st.take_dirty_list()
        };
        let aplus_copy = { quic.state.lock().unwrap().aplus_layers.clone() };
        let (_, ms) = world.update_dirty_chunks(&aplus_copy, &dirty_init);
        ms
    };

    // Observe 10 Hz incremental 3D dirty-chunk / dirty-brick rebuilds
    let mut dirty_samples = Vec::new();
    let t_obs = Instant::now();
    let mut last_tick = 0u64;
    let mut test_chat_sent = false;
    while t_obs.elapsed().as_secs_f64() < args.observe_secs {
        if !test_chat_sent && t_obs.elapsed().as_secs_f64() >= 1.5 {
            if let Some(ref text) = args.test_chat {
                let _ = rt.block_on(QuicClientHandle::send_reliable_on(
                    &quic.conn,
                    &ClientReliable::ChatCommand {
                        text: text.clone(),
                        volume: 1.0,
                    },
                ));
                println!("[Unified] Sent headless --test-chat: {:?} (SHOUT 1.0)", text);
            }
            test_chat_sent = true;
        }
        if let Some(ref mut z16w) = z16_world {
            let mut st = quic.state.lock().unwrap();
            if st.depth > 32 {
                let dirty_subs = st.take_dirty_z16_subbricks();
                if !dirty_subs.is_empty() {
                    let (cnt, ms) = z16w.apply_dirty_subbricks(
                        &st.z16_density_mat,
                        &st.z16_albedo_thermal,
                        &dirty_subs,
                    );
                    if cnt > 0 {
                        dirty_samples.push(ms);
                    }
                }
            } else {
                let dirty_z16 = st.take_dirty_z16_bricks();
                if !dirty_z16.is_empty() {
                    let (cnt, ms) = z16w.apply_dirty_bricks(
                        &st.z16_density_mat,
                        &st.z16_albedo_thermal,
                        &dirty_z16,
                    );
                    if cnt > 0 {
                        dirty_samples.push(ms);
                    }
                }
            }
        } else {
            let (ft, dirty, snap_layers) = {
                let mut st = quic.state.lock().unwrap();
                let d = st.take_dirty_list();
                let cells = if !d.is_empty() { Some(st.aplus_layers.clone()) } else { None };
                (st.latest_field_tick, d, cells)
            };
            if let Some(layers) = snap_layers {
                if ft > last_tick {
                    last_tick = ft;
                    let (cnt, ms) = world.update_dirty_chunks(&layers, &dirty);
                    if cnt > 0 {
                        dirty_samples.push(ms);
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(12));
    }
    dirty_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dirty_p50 = dirty_samples.get(dirty_samples.len() / 2).copied().unwrap_or(0.0);
    let dirty_p99 = dirty_samples
        .get(((dirty_samples.len() as f64) * 0.99).floor().min(dirty_samples.len().saturating_sub(1) as f64) as usize)
        .copied()
        .unwrap_or(dirty_p50);

    let entities = { quic.state.lock().unwrap().entities.clone() };
    let atmo = entities
        .as_ref()
        .map(|e| Atmosphere {
            weather_kind: e.telemetry.weather_kind,
            day_time: e.telemetry.day_time,
            wind_x: e.telemetry.wind_x,
            wind_y: e.telemetry.wind_y,
        })
        .unwrap_or_default();

    renderer.set_z16_mode(z16_active, 0);
    renderer.set_z16_hero_cutaway(Vec3::ZERO, 0.0);
    if let Some(ref mut z16w) = z16_world {
        renderer.upload_z16_world(z16w);
        renderer.upload_minimap(z16w.width as u32, z16w.height as u32, &z16w.minimap_rgba);
    } else {
        renderer.upload_minimap(world.grid_w as u32, world.grid_h as u32, &world.minimap_rgba);
    }

    // Heroic 3D Volumetric Vista:
    // - Z=64 mode: grand 3-strata framing capturing Floating Sky-Islands + 28m plunging waterfalls +
    //   Surface Viaduct + Abyssal Chasm simultaneously!
    // - Z=16..32 mode: frames the 3-Span Arched Stone & Timber Bridge, river canyon, North alpine waterfall,
    //   SW cave massif, Hard-Z sculpted pines, and live multiplayer heroes!
    // - L4 mode: frames Layer 2 Highland Plateaus, Plunging 3D Waterfalls, Stone Sky-Arch & Central Spire!
    let cam_main = if z16_active && gd >= 48 {
        MobaCamera {
            target: Vec3::new(0.0, gd as f32 * 0.44, 0.0),
            yaw: 0.52,
            pitch: 0.28,
            dist: (gw as f32 * 0.52).clamp(96.0, 138.0),
            fov_y: 44.0_f32.to_radians(),
            ortho: false,
        }
    } else if z16_active {
        let target_y = (gd as f32 * 0.35).clamp(5.6, 11.5);
        MobaCamera {
            target: Vec3::new(0.0, target_y, -4.0),
            yaw: 0.46,
            pitch: 0.42,
            dist: (gw as f32 * 0.46).clamp(48.0, 96.0),
            fov_y: 38.0_f32.to_radians(),
            ortho: false,
        }
    } else {
        MobaCamera {
            target: Vec3::new(0.0, 2.2, -(gh as f32) * 0.14),
            yaw: 0.48,
            pitch: 0.56,
            dist: 102.0,
            fov_y: 38.0_f32.to_radians(),
            ortho: false,
        }
    };
    let sun_dir = Approach4Renderer::sun_dir(atmo);
    let mats_main = cam_main.compute_matrices(1920.0, 1080.0, sun_dir);
    let sim_time = 2.4f32;
    let frame_main = assemble_live_frame(
        &world,
        z16_world.as_ref(),
        entities.as_ref(),
        &surfels,
        &mats_main,
        cam_main.target,
        cam_main.dist,
        sim_time,
        0,
        None,
        true,
    );

    // Benchmark 4 modes (with full HUD active!)
    let modes = [
        ("1080p Pixel-Art + Vol3DGS + HUD", 1920u32, 1080u32, RenderStyle::pixel_art(2)),
        ("1080p Clean 3D + Vol3DGS + HUD", 1920, 1080, RenderStyle::clean_3d()),
        ("4K Pixel-Art + Vol3DGS + HUD", 3840, 2160, RenderStyle::pixel_art(4)),
        ("4K Clean 3D + Vol3DGS + HUD", 3840, 2160, RenderStyle::clean_3d()),
    ];

    let mut gpu_benchmarks = Vec::new();
    let mut fps_1080 = 600.0f32;
    for (label, bw, bh, st) in &modes {
        let m = cam_main.compute_matrices(*bw as f32, *bh as f32, sun_dir);
        let hb = build_hud(
            *bw as f32,
            *bh as f32,
            &world,
            z16_world.as_ref(),
            entities.as_ref(),
            &m,
            &HudConfig {
                show_hud: true,
                show_telemetry: true,
                show_brain_inspector: true,
                brain_selected_idx: None,
                selected_element: 0,
                focus_radius: 4.0,
                style_label: st.label,
                z16_active,
                z16_style_label: z16_style_label(0),
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
                cave_enclosure: 0.0,
                show_squad_links: true,
                camera_preset_label: CameraPreset::Tactical.label(),
                macro_eye_f64: None,
            },
        );
        let (p50, p99, mean) = renderer.benchmark_gpu(
            *bw,
            *bh,
            st,
            atmo,
            &cam_main,
            &m,
            &frame_main,
            Some(&hb),
            8,
            args.bench_iters,
        );
        let fps = 1000.0 / p50.max(0.001);
        if *bw == 1920 && st.crisp_resolve {
            fps_1080 = fps as f32;
        }
        let art_w = bw / st.pixel_scale.max(1);
        let art_h = bh / st.pixel_scale.max(1);
        println!(
            "  [Bench] {:34} | p50={:.3}ms p99={:.3}ms ({:.1} FPS)",
            label, p50, p99, fps
        );
        gpu_benchmarks.push(GpuBenchEntry {
            label: (*label).to_string(),
            resolution: format!("{}x{}", bw, bh),
            art_resolution: format!("{}x{}", art_w, art_h),
            style: st.label.to_string(),
            p50_ms: (p50 * 1000.0).round() / 1000.0,
            p99_ms: (p99 * 1000.0).round() / 1000.0,
            mean_ms: (mean * 1000.0).round() / 1000.0,
            fps_p50: (fps * 10.0).round() / 10.0,
        });
    }

    // Capture 1: Crisp 1920×1080 Live Battle + Full HUD + Live Tactical Minimap
    let style_main = style_from_name(&args.style, 1080);
    let hud_main = build_hud(
        1920.0,
        1080.0,
        &world,
        z16_world.as_ref(),
        entities.as_ref(),
        &mats_main,
        &HudConfig {
            show_hud: true,
            show_telemetry: true,
            show_brain_inspector: true,
            brain_selected_idx: None,
            selected_element: 0,
            focus_radius: 4.0,
            style_label: style_main.label,
            z16_active,
            z16_style_label: z16_style_label(0),
            fps: fps_1080,
            rtt_ms: 1.2,
            chat_open: false,
            chat_input: "",
            chat_shout: true,
            show_scoreboard: false,
            audio_muted: true,
            audio_available: false,
            cave_enclosure: 0.0,
            show_squad_links: true,
            camera_preset_label: CameraPreset::Tactical.label(),
            macro_eye_f64: None,
        },
    );
    let (rgba_main, _, _) = renderer.render_to_rgba(
        1920,
        1080,
        &style_main,
        atmo,
        &cam_main,
        &mats_main,
        sim_time,
        &frame_main,
        Some(&hud_main),
    );
    std::fs::create_dir_all(&args.out_dir)?;
    save_rgba_png(&args.out_dir.join("unified_live_1080p_hud.png"), 1920, 1080, &rgba_main)?;

    // Capture 2: 2×2 Live WGSL 4-Style Neural Compute Showcase (960×540 per panel -> 1920×1080)
    let views: [(MobaCamera, RenderStyle, bool, u8, u32); 4] = if z16_active {
        let (base_d, ty, base_pitch, base_fov) = if gd >= 48 {
            ((gw as f32 * 0.50).clamp(92.0, 134.0), gd as f32 * 0.44, 0.34, 44.0_f32.to_radians())
        } else {
            ((gw as f32 * 0.44).clamp(46.0, 92.0), (gd as f32 * 0.35).clamp(5.6, 11.5), 0.42, 38.0_f32.to_radians())
        };
        [
            // Top-left: Style 0 — NORDIC (PIXEL-2XSS) + Full HUD
            (
                MobaCamera { target: Vec3::new(0.0, ty, -4.0), yaw: 0.48, pitch: base_pitch, dist: base_d, fov_y: base_fov, ortho: false },
                RenderStyle::pixel_art(1),
                true,
                0u8,
                0u32,
            ),
            // Top-right: Style 1 — VOLCANIC EMBER (PIXEL-2XSS)
            (
                MobaCamera { target: Vec3::new(0.0, ty, -4.0), yaw: -0.52, pitch: base_pitch, dist: base_d, fov_y: base_fov, ortho: false },
                RenderStyle::pixel_art(1),
                false,
                0u8,
                1u32,
            ),
            // Bottom-left: Style 2 — ARCANE THERMAL (CLEAN-3D)
            (
                MobaCamera { target: Vec3::new(0.0, ty, -4.0), yaw: 0.62, pitch: (base_pitch + 0.04).min(0.48), dist: base_d, fov_y: base_fov, ortho: false },
                RenderStyle::clean_3d(),
                false,
                0u8,
                2u32,
            ),
            // Bottom-right: Style 3 — EMERALD CAVERN (Low dramatic angle framing Viaduct arches & Abyssal Chasm!)
            (
                MobaCamera { target: Vec3::new(-4.0, (gd as f32 * 0.28).clamp(4.6, 22.0), 0.0), yaw: 0.34, pitch: 0.22, dist: base_d * 0.54, fov_y: 38.0_f32.to_radians(), ortho: false },
                RenderStyle::pixel_art(1),
                false,
                0u8,
                3u32,
            ),
        ]
    } else {
        [
            (
                MobaCamera { target: Vec3::new(0.0, 2.0, -(gh as f32) * 0.14), yaw: 0.44, pitch: 0.62, dist: 124.0, fov_y: 40.0_f32.to_radians(), ortho: false },
                RenderStyle::pixel_art(1),
                true,
                0u8,
                0u32,
            ),
            (
                MobaCamera { target: Vec3::new(-4.0, 2.5, -(gh as f32) * 0.23), yaw: 0.35, pitch: 0.36, dist: 56.0, fov_y: 36.0_f32.to_radians(), ortho: false },
                RenderStyle::pixel_art(1),
                false,
                0u8,
                0u32,
            ),
            (
                MobaCamera { target: Vec3::new(0.0, 4.0, -(gh as f32) * 0.12), yaw: 0.62, pitch: 0.56, dist: 112.0, fov_y: 38.0_f32.to_radians(), ortho: false },
                RenderStyle::pixel_art(1),
                false,
                2u8,
                0u32,
            ),
            (
                MobaCamera { target: Vec3::new(14.0, 2.5, (gh as f32) * 0.18), yaw: -0.95, pitch: 0.46, dist: 68.0, fov_y: 38.0_f32.to_radians(), ortho: false },
                RenderStyle::clean_3d(),
                false,
                0u8,
                0u32,
            ),
        ]
    };
    let mut panels = Vec::with_capacity(4);
    for (cam, st, with_hud, slice_mode, z16_st) in &views {
        renderer.set_z16_mode(z16_active, *z16_st);
        let m = cam.compute_matrices(960.0, 540.0, sun_dir);
        let fr = assemble_live_frame(
            &world,
            z16_world.as_ref(),
            entities.as_ref(),
            &surfels,
            &m,
            cam.target,
            cam.dist,
            sim_time,
            *slice_mode,
            None,
            true,
        );
        let hb = if *with_hud {
            Some(build_hud(
                960.0,
                540.0,
                &world,
                z16_world.as_ref(),
                entities.as_ref(),
                &m,
                &HudConfig {
                    show_hud: true,
                    show_telemetry: true,
                    show_brain_inspector: true,
                    brain_selected_idx: None,
                    selected_element: 1,
                    focus_radius: 4.0,
                    style_label: st.label,
                    z16_active,
                    z16_style_label: z16_style_label(*z16_st),
                    fps: fps_1080,
                    rtt_ms: 1.2,
                    chat_open: false,
                    chat_input: "",
                    chat_shout: true,
                    show_scoreboard: false,
                    audio_muted: true,
                    audio_available: false,
                    cave_enclosure: 0.0,
                    show_squad_links: true,
                    camera_preset_label: CameraPreset::Tactical.label(),
                    macro_eye_f64: None,
                },
            ))
        } else {
            None
        };
        let (px, _, _) = renderer.render_to_rgba(960, 540, st, atmo, cam, &m, sim_time, &fr, hb.as_ref());
        panels.push(px);
    }
    renderer.set_z16_mode(z16_active, 0);
    let grid = compose_2x2_grid([&panels[0], &panels[1], &panels[2], &panels[3]], 960, 540);
    save_rgba_png(&args.out_dir.join("unified_orbit_4view.png"), 1920, 1080, &grid)?;

    // Capture 3: Dramatic Close-Up 3/4 Action View (`unified_closeup_action_1080p.png`, dist ~24m)
    // Automatically centers on the densest active multi-archetype skirmish near the bridge & river!
    let closeup_target = {
        let default_ty = if z16_active { (gd as f32 * 0.34).clamp(5.4, 10.5) } else { 2.0 };
        let mut best_pos = Vec3::new(0.0, default_ty, -2.0);
        let mut best_score = -1.0f32;
        if let Some(ref ent) = entities {
            let to_wpos = |gx: f32, gy: f32, gz: f32| -> Vec3 {
                if let Some(ref z16w) = z16_world {
                    let (wx, wz) = z16w.grid_to_world_xz(gx, gy);
                    Vec3::new(wx, gz.max(z16w.sample_surface_y(gx, gy)), wz)
                } else {
                    let (wx, wz) = world.grid_to_world_xz(gx, gy);
                    Vec3::new(wx, world.sample_3d_y(gx, gy, gz), wz)
                }
            };
            for p in &ent.players {
                if p.hp <= 0 {
                    continue;
                }
                let wp = to_wpos(p.x, p.y, p.z);
                let mut arch_mask = 1u8 << (p.class_id.min(4));
                let mut team_mask = 1u8 << (p.team.min(1));
                let mut neighbors = 0.0f32;
                let mut centroid = wp;
                let mut w_sum = 1.0f32;
                for q in &ent.players {
                    if q.id == p.id || q.hp <= 0 {
                        continue;
                    }
                    let qp = to_wpos(q.x, q.y, q.z);
                    let d = wp.distance(qp);
                    if d < 18.0 {
                        let w = (1.0 - d / 18.0).max(0.1);
                        neighbors += w;
                        centroid += qp * w;
                        w_sum += w;
                        arch_mask |= 1u8 << (q.class_id.min(4));
                        team_mask |= 1u8 << (q.team.min(1));
                    }
                }
                let mut pulse_bonus = 0.0f32;
                for pulse in &ent.actuator_pulses {
                    let (px, pz) = if let Some(ref z16w) = z16_world {
                        z16w.grid_to_world_xz((pulse.x0 + pulse.x1) * 0.5, (pulse.y0 + pulse.y1) * 0.5)
                    } else {
                        world.grid_to_world_xz((pulse.x0 + pulse.x1) * 0.5, (pulse.y0 + pulse.y1) * 0.5)
                    };
                    if (px - wp.x).hypot(pz - wp.z) < 20.0 {
                        pulse_bonus += 2.5;
                    }
                }
                // Strongly favor skirmishes on/near the iconic central arched bridge, Spire & fjord river!
                let dist_center = wp.x.hypot(wp.z + 2.0);
                let bridge_prox = (1.0 - (dist_center / 34.0)).clamp(0.0, 1.0);
                let score = neighbors * 1.35
                    + (arch_mask.count_ones() as f32) * 1.6
                    + if team_mask == 0b11 { 3.2 } else { 0.0 }
                    + pulse_bonus
                    + bridge_prox * 14.0;
                if score > best_score {
                    best_score = score;
                    let c = centroid / w_sum;
                    best_pos = Vec3::new(c.x, c.y + 1.0, c.z);
                }
            }
        }
        best_pos
    };
    let cam_closeup = MobaCamera {
        target: closeup_target,
        yaw: 0.46,
        pitch: 0.34,
        dist: 26.0,
        fov_y: 40.0_f32.to_radians(),
        ortho: false,
    };
    let mats_closeup = cam_closeup.compute_matrices(1920.0, 1080.0, sun_dir);
    let frame_closeup = assemble_live_frame(
        &world,
        z16_world.as_ref(),
        entities.as_ref(),
        &surfels,
        &mats_closeup,
        cam_closeup.target,
        cam_closeup.dist,
        sim_time,
        0,
        None,
        true,
    );
    let hud_closeup = build_hud(
        1920.0,
        1080.0,
        &world,
        z16_world.as_ref(),
        entities.as_ref(),
        &mats_closeup,
        &HudConfig {
            show_hud: true,
            show_telemetry: true,
            show_brain_inspector: true,
            brain_selected_idx: None,
            selected_element: 0,
            focus_radius: 4.0,
            style_label: style_main.label,
            z16_active,
            z16_style_label: z16_style_label(0),
            fps: fps_1080,
            rtt_ms: 1.2,
            chat_open: false,
            chat_input: "",
            chat_shout: true,
            show_scoreboard: false,
            audio_muted: true,
            audio_available: false,
            cave_enclosure: 0.0,
            show_squad_links: true,
            camera_preset_label: CameraPreset::ActionCloseUp.label(),
            macro_eye_f64: None,
        },
    );
    let (rgba_closeup, _, _) = renderer.render_to_rgba(
        1920,
        1080,
        &style_main,
        atmo,
        &cam_closeup,
        &mats_closeup,
        sim_time,
        &frame_closeup,
        Some(&hud_closeup),
    );
    save_rgba_png(&args.out_dir.join("unified_closeup_action_1080p.png"), 1920, 1080, &rgba_closeup)?;

    let tel = entities.as_ref().map(|e| e.telemetry.clone());
    let mesh_inst_cnt: usize = frame_main.mesh_batches.iter().map(|b| b.len()).sum();
    let terrain_tris = z16_world
        .as_ref()
        .map(|z| z.total_triangles())
        .unwrap_or_else(|| frame_main.terrain_idxs.len() / 3);
    let metrics = UnifiedMetrics {
        server: args.server.to_string(),
        gpu_adapter: renderer.adapter_name.clone(),
        gpu_backend: renderer.adapter_backend.clone(),
        grid_w: world.grid_w,
        grid_h: world.grid_h,
        total_chunks: world.chunks.len(),
        parametric_build_ms: (parametric_build_ms * 100.0).round() / 100.0,
        sim_mode: tel.as_ref().map(|t| t.mode.clone()).unwrap_or_default(),
        backend: tel.as_ref().map(|t| t.backend.clone()).unwrap_or_default(),
        field_tick: tel.as_ref().map(|t| t.field_tick).unwrap_or(0),
        world_infer_ms: tel.as_ref().map(|t| t.infer_ms_last).unwrap_or(0.0),
        eco_infer_ms: tel.as_ref().map(|t| t.eco_infer_ms).unwrap_or(0.0),
        proj_active_pct: tel.as_ref().map(|t| t.proj_active_pct).unwrap_or(0.0),
        mass_error: tel.as_ref().map(|t| t.mass_error).unwrap_or(0),
        energy_error: tel.as_ref().map(|t| t.energy_error).unwrap_or(0),
        wire_kbps: tel.as_ref().map(|t| t.wire_kbps).unwrap_or(0.0),
        aoi_chunks_sent: tel.as_ref().map(|t| t.aoi_chunks_sent).unwrap_or(0),
        active_bots: tel.as_ref().map(|t| t.active_bots).unwrap_or(0),
        weather_kind: tel.as_ref().map(|t| t.weather_kind).unwrap_or(0),
        dirty_update_p50_ms: (dirty_p50 * 1000.0).round() / 1000.0,
        dirty_update_p99_ms: (dirty_p99 * 1000.0).round() / 1000.0,
        full_sync_build_ms: (full_sync_build_ms * 1000.0).round() / 1000.0,
        visible_chunks: frame_main.visible_chunks,
        terrain_triangles: terrain_tris,
        mesh_instances: mesh_inst_cnt,
        surfel_draws: frame_main.surfel_refs.len(),
        volumetric_splats: frame_main.vol_splats.len(),
        hud_vertices: hud_main.vertices.len(),
        gpu_benchmarks,
    };
    let json = serde_json::to_string_pretty(&metrics)?;
    std::fs::write(args.out_dir.join("unified_metrics.json"), &json)?;
    println!("[Unified] Saved unified_live_1080p_hud.png, unified_orbit_4view.png, unified_closeup_action_1080p.png, unified_metrics.json");
    Ok(())
}

// =========================================================================
// INTERACTIVE PLAYABLE WINDOW MODE (macOS Metal / Linux Vulkan via winit)
// =========================================================================

#[derive(Clone, Copy, Debug)]
struct PendingMoveInput {
    seq: u32,
    move_x: f32,
    move_y: f32,
    dt: f32,
    dash: bool,
}

fn hero_class_base_speed(class_id: u8) -> f32 {
    match class_id {
        CLASS_WARDEN => 15.5,    // Strider (Heavy Walker / Squad Anchor)
        CLASS_ALCHEMIST => 16.0, // Skywing (3D Flyer / Thermal Glider)
        CLASS_RANGER => 18.8,    // Tidekin (Amphibious Swimmer / Diver)
        3 => 19.2,               // Emberstalker (Solo Cavern Climber / Raider)
        CLASS_BURROWER => 16.4,  // Burrower (Subterranean Crystal-Plated Golem)
        _ => 19.2,
    }
}

/// Discrete per-cell solid density matching `frostmarsh-server` & `world_bridge.rs` 3×3 spaced pine trunks 1:1.
#[inline]
fn client_cell_solid_density(
    world: &LiveApproach4World,
    xi: i32,
    yi: i32,
    on_l2: bool,
    allow_water: bool,
) -> f32 {
    let x = xi.clamp(0, (world.grid_w.saturating_sub(1)) as i32) as usize;
    let y = yi.clamp(0, (world.grid_h.saturating_sub(1)) as i32) as usize;
    let idx = y * world.grid_w + x;
    if on_l2 {
        let c2 = world.decoded_l2[idx];
        if c2.s1 == SP_ROCK || (c2.wall && c2.s1 != SP_WOOD) {
            return 1.0;
        }
        if c2.wall && c2.s1 == SP_WOOD && !c2.walkable {
            return if x % 3 == 0 && y % 3 == 0 { 1.0 } else { 0.0 };
        }
        if c2.walkable && !c2.wall {
            return 0.0;
        }
        let c1 = world.decoded[idx];
        return if c1.s1 == SP_ROCK || (c1.wall && c1.s1 != SP_WOOD) { 1.0 } else { 0.0 };
    }
    let c1 = world.decoded[idx];
    if c1.s1 == SP_ROCK || (c1.wall && c1.s1 != SP_WOOD) {
        return 1.0;
    }
    if c1.wall && c1.s1 == SP_WOOD && !c1.walkable {
        return if x % 3 == 0 && y % 3 == 0 { 1.0 } else { 0.0 };
    }
    if !allow_water && c1.water_slow && !c1.walkable && !c1.wall {
        return 0.85;
    }
    0.0
}

/// Continuous bilinear solid-density field at `(x, y, on_l2)` (exact 1:1 parity with `frostmarsh-server`).
#[inline]
fn sample_solid_density(
    world: &LiveApproach4World,
    x: f32,
    y: f32,
    on_l2: bool,
    allow_water: bool,
) -> f32 {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = (fx - x0 as f32).clamp(0.0, 1.0);
    let ty = (fy - y0 as f32).clamp(0.0, 1.0);
    let d00 = client_cell_solid_density(world, x0, y0, on_l2, allow_water);
    let d10 = client_cell_solid_density(world, x0 + 1, y0, on_l2, allow_water);
    let d01 = client_cell_solid_density(world, x0, y0 + 1, on_l2, allow_water);
    let d11 = client_cell_solid_density(world, x0 + 1, y0 + 1, on_l2, allow_water);
    let d0 = d00 + (d10 - d00) * tx;
    let d1 = d01 + (d11 - d01) * tx;
    d0 + (d1 - d0) * ty
}

/// Hero capsule solid density (`R = 0.65`, center + 4 perimeter probes at `0.55`).
#[inline]
fn sample_capsule_density(
    world: &LiveApproach4World,
    x: f32,
    y: f32,
    on_l2: bool,
    allow_water: bool,
) -> f32 {
    const PROBE_R: f32 = 0.55;
    let c0 = sample_solid_density(world, x, y, on_l2, allow_water);
    let pe = sample_solid_density(world, x + PROBE_R, y, on_l2, allow_water);
    let pw = sample_solid_density(world, x - PROBE_R, y, on_l2, allow_water);
    let ps = sample_solid_density(world, x, y + PROBE_R, on_l2, allow_water);
    let pn = sample_solid_density(world, x, y - PROBE_R, on_l2, allow_water);
    0.40 * c0 + 0.15 * (pe + pw + ps + pn)
}

/// Exact client-side mirror of `frostmarsh-server`'s continuous bilinear capsule wall-sliding locomotion
/// so predicted hero motion matches authoritative server updates with zero snagging and zero rubber-banding!
#[allow(clippy::too_many_arguments)]
fn predict_hero_step(
    world: &LiveApproach4World,
    x: f32,
    y: f32,
    z: f32,
    move_x: f32,
    move_y: f32,
    dt: f32,
    class_id: u8,
    carrying_relic: bool,
    dashing: bool,
    hasted: bool,
    bodies: &[CellBodySnap],
) -> (f32, f32) {
    let gw = world.grid_w as f32;
    let gh = world.grid_h as f32;
    let mlen = move_x.hypot(move_y);
    let (dx, dy) = if mlen >= 0.01 { (move_x / mlen, move_y / mlen) } else { (0.0, 0.0) };
    let mut speed = if dashing {
        42.0
    } else if mlen < 0.01 {
        0.0
    } else {
        let mut s = hero_class_base_speed(class_id);
        if carrying_relic { s *= 0.78; }
        if hasted { s *= 1.32; }
        s
    };

    let on_l2 = z >= 1.55;
    let sample_cell = |tx: f32, ty: f32| {
        let xi = (tx.floor() as i32).clamp(0, (world.grid_w.saturating_sub(1)) as i32) as usize;
        let yi = (ty.floor() as i32).clamp(0, (world.grid_h.saturating_sub(1)) as i32) as usize;
        let idx = yi * world.grid_w + xi;
        if on_l2 { world.decoded_l2[idx] } else { world.decoded[idx] }
    };

    let c_here = sample_cell(x, y);
    let on_raft = z < 1.35
        && bodies.iter().any(|b| (x - b.x).powi(2) + (y - b.y).powi(2) <= (b.radius + 1.0).powi(2));
    if !dashing && speed > 0.0 {
        if !on_raft && c_here.water_slow && c_here.s1 != SP_ICE {
            speed *= 0.60;
        } else if c_here.s1 == SP_ICE {
            speed *= 1.20;
        }
    }

    let mut vx_s = dx * speed;
    let mut vy_s = dy * speed;
    let capsule_at = |tx: f32, ty: f32| -> f32 {
        let allow_water = dashing
            || on_raft
            || (z < 1.35
                && bodies.iter().any(|b| {
                    (tx - b.x).powi(2) + (ty - b.y).powi(2) <= (b.radius + 1.0).powi(2)
                }));
        sample_capsule_density(world, tx, ty, on_l2, allow_water)
    };

    let mut fx = (x + vx_s * dt).clamp(3.5, (gw - 3.5).max(3.5));
    let mut fy = (y + vy_s * dt).clamp(3.5, (gh - 3.5).max(3.5));
    for _ in 0..2 {
        let dens = capsule_at(fx, fy);
        if dens <= 0.35 {
            break;
        }
        let eps = 0.35f32;
        let gx = (capsule_at(fx + eps, fy) - capsule_at(fx - eps, fy)) / (2.0 * eps);
        let gy = (capsule_at(fx, fy + eps) - capsule_at(fx, fy - eps)) / (2.0 * eps);
        let gmag = gx.hypot(gy);
        if gmag > 1e-4 {
            let nx = gx / gmag;
            let ny = gy / gmag;
            let vn = vx_s * nx + vy_s * ny;
            if vn > 0.0 {
                vx_s -= vn * nx;
                vy_s -= vn * ny;
            }
            let push = ((dens - 0.35) * 0.65).min(0.45);
            fx = (x + vx_s * dt - nx * push).clamp(3.5, (gw - 3.5).max(3.5));
            fy = (y + vy_s * dt - ny * push).clamp(3.5, (gh - 3.5).max(3.5));
        } else {
            fx = x;
            fy = y;
            break;
        }
    }
    let d_final = capsule_at(fx, fy);
    if d_final > 0.62 && capsule_at(x, y) < d_final {
        fx = x;
        fy = y;
    }
    (fx, fy)
}

struct InteractiveApp {
    args: Args,
    rt: tokio::runtime::Runtime,
    quic: Option<QuicClientHandle>,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_cfg: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<Approach4Renderer>,
    meshes: StaticMeshBank,
    surfels: SurfelProtoBank,
    world: LiveApproach4World,
    z16_world: Z16World,
    z16_active: bool,
    planet_world: PlanetWorld,
    planet_mode: bool,
    planet_camera: PlanetCamera,
    macro_planet_world: MacroPlanetWorld,
    macro_planet_mode: bool,
    macro_planet_camera: MacroPlanetCamera,
    z16_style_mode: u32,
    z16_cam_framed: bool,
    cave_enclosure_ema: f32,
    cutaway_weight: f32,
    camera: MobaCamera,
    camera_preset: CameraPreset,
    camera_preset_easing: bool,
    style_clean: bool,
    layer_slice_mode: u8,
    show_hud: bool,
    show_telemetry: bool,
    show_squad_links: bool,
    brain_inspector_on: bool,
    brain_selected_idx: Option<usize>,
    selected_element: u8,
    focus_radius: f32,
    next_dome_mode: u8,
    last_cursor_grid: Option<(f32, f32)>,
    chat_open: bool,
    chat_input: String,
    chat_shout: bool,
    audio: Option<SpatialAudioEngine>,
    audio_muted: bool,
    seen_proj_ids: HashSet<u32>,
    seen_event_ids: VecDeque<u32>,
    seen_bubble_ticks: HashMap<u32, u32>,
    last_audio_sim_tick: u64,
    keys: HashSet<KeyCode>,
    mouse_pos: (f32, f32),
    mouse_left: bool,
    mouse_right: bool,
    mouse_mid: bool,
    last_mouse_drag: Option<(f32, f32)>,
    pending_dash: bool,
    pending_carve: bool,
    pending_ability_e: bool,
    pending_ult_r: bool,
    input_seq: u32,
    input_dirty: bool,
    actuator_edge_fire: bool,
    last_input_send: Instant,
    last_actuator_send: Instant,
    start_time: Instant,
    last_frame: Instant,
    fps_ema: f32,
    smooth_players: HashMap<u32, (f32, f32, f32)>,
    smooth_projs: HashMap<u32, (f32, f32, f32)>,
    smooth_bodies: HashMap<u32, (f32, f32, f32)>,
    pending_inputs: VecDeque<PendingMoveInput>,
    pred_hero_pos: Option<(f32, f32, f32)>,
    pred_hero_vz: f32,
    hero_visual_err: (f32, f32),
    hero_dash_pred_until: Option<Instant>,
    local_aim_dir: (f32, f32),
    local_actuator_pulses: Vec<(Instant, ActuatorPulseSnap)>,
    last_snap_tick: u64,
    snap_age_s: f32,
}

impl InteractiveApp {
    fn new(args: Args) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio rt");
        let meshes = StaticMeshBank::build(1);
        let surfels = SurfelProtoBank::build();
        let style_clean = args.style == "clean";
        let audio = SpatialAudioEngine::try_new();
        Self {
            args,
            rt,
            quic: None,
            window: None,
            surface: None,
            surface_cfg: None,
            renderer: None,
            meshes,
            surfels,
            world: LiveApproach4World::new(GRID_W, GRID_H, GRID_D),
            z16_world: Z16World::new(128, 128, 16),
            z16_active: false,
            planet_world: PlanetWorld::new(64, 32, 48.0),
            planet_mode: false,
            planet_camera: PlanetCamera::new(48.0, 32.0),
            macro_planet_world: MacroPlanetWorld::new(1280.0, 8, 64, 33_033),
            macro_planet_mode: false,
            macro_planet_camera: MacroPlanetCamera::new(1304.0),
            z16_style_mode: 0,
            z16_cam_framed: false,
            cave_enclosure_ema: 0.0,
            cutaway_weight: 0.0,
            camera: MobaCamera {
                target: Vec3::new(0.0, 1.6, 0.0),
                yaw: 0.42,
                pitch: 0.64,
                dist: 44.0,
                fov_y: 38.0_f32.to_radians(),
                ortho: false,
            },
            camera_preset: CameraPreset::Tactical,
            camera_preset_easing: false,
            style_clean,
            layer_slice_mode: 0,
            show_hud: true,
            show_telemetry: true,
            show_squad_links: true,
            brain_inspector_on: true,
            brain_selected_idx: None,
            selected_element: 0,
            focus_radius: 4.0,
            next_dome_mode: 1,
            last_cursor_grid: None,
            chat_open: false,
            chat_input: String::new(),
            chat_shout: true,
            audio,
            audio_muted: false,
            seen_proj_ids: HashSet::new(),
            seen_event_ids: VecDeque::with_capacity(64),
            seen_bubble_ticks: HashMap::new(),
            last_audio_sim_tick: 0,
            keys: HashSet::new(),
            mouse_pos: (720.0, 450.0),
            mouse_left: false,
            mouse_right: false,
            mouse_mid: false,
            last_mouse_drag: None,
            pending_dash: false,
            pending_carve: false,
            pending_ability_e: false,
            pending_ult_r: false,
            input_seq: 0,
            input_dirty: false,
            actuator_edge_fire: false,
            last_input_send: Instant::now(),
            last_actuator_send: Instant::now(),
            start_time: Instant::now(),
            last_frame: Instant::now(),
            fps_ema: 60.0,
            smooth_players: HashMap::new(),
            smooth_projs: HashMap::new(),
            smooth_bodies: HashMap::new(),
            pending_inputs: VecDeque::with_capacity(96),
            pred_hero_pos: None,
            pred_hero_vz: 0.0,
            hero_visual_err: (0.0, 0.0),
            hero_dash_pred_until: None,
            local_aim_dir: (0.0, -1.0),
            local_actuator_pulses: Vec::new(),
            last_snap_tick: 0,
            snap_age_s: 0.0,
        }
    }

    fn render_frame(&mut self) {
        let dt = self.last_frame.elapsed().as_secs_f32().clamp(0.0005, 0.1);
        self.last_frame = Instant::now();
        let inst_fps = 1.0 / dt;
        self.fps_ema = self.fps_ema * 0.90 + inst_fps * 0.10;
        let sim_time = self.start_time.elapsed().as_secs_f32();

        // Camera keyboard orbit + smooth F5 CameraPreset easing (locked while typing in acoustic chat!)
        if !self.chat_open {
            if self.keys.contains(&KeyCode::KeyQ) || self.keys.contains(&KeyCode::ArrowLeft) {
                self.camera.yaw -= 1.6 * dt;
                self.planet_camera.orbit_yaw -= 1.8 * dt;
                self.macro_planet_camera.orbit_yaw -= 1.8 * dt as f64;
            }
            if self.keys.contains(&KeyCode::KeyZ) || self.keys.contains(&KeyCode::ArrowRight) {
                self.camera.yaw += 1.6 * dt;
                self.planet_camera.orbit_yaw += 1.8 * dt;
                self.macro_planet_camera.orbit_yaw += 1.8 * dt as f64;
            }
        }
        if self.camera_preset_easing {
            let (tgt_dist, tgt_pitch, tgt_fov) =
                self.camera_preset.params_for_depth(self.z16_active, self.z16_world.depth);
            let ease = 1.0 - (-9.5 * dt).exp();
            self.camera.dist += (tgt_dist - self.camera.dist) * ease;
            self.camera.pitch += (tgt_pitch - self.camera.pitch) * ease;
            self.camera.fov_y += (tgt_fov - self.camera.fov_y) * ease;
            if (self.camera.dist - tgt_dist).abs() < 0.08
                && (self.camera.pitch - tgt_pitch).abs() < 0.004
                && (self.camera.fov_y - tgt_fov).abs() < 0.002
            {
                self.camera.dist = tgt_dist;
                self.camera.pitch = tgt_pitch;
                self.camera.fov_y = tgt_fov;
                self.camera_preset_easing = false;
            }
        }

        // Pull latest 3D volumetric dirty bricks/sub-bricks (Z=16..64) OR 6-Face Planet bricks OR A+ dirty chunks (L4) + entities from QUIC
        let (mut entities, rtt_ms) = if let Some(ref quic) = self.quic {
            let mut st = quic.state.lock().unwrap();
            let gw = st.width;
            let gh = st.height;
            let gd = st.depth;
            let ent = st.entities.clone();
            let rtt = st.rtt_ms;
            let is_macro_planet = st.macro_planet_mode
                || st.received_macro_shard_brick_count() > 0
                || st.received_macro_climate_face_count() > 0
                || ent.as_ref().is_some_and(|e| e.telemetry.macro_planet);
            let is_planet = !is_macro_planet
                && (st.planet_mode
                    || st.received_planet_brick_count() > 0
                    || ent.as_ref().is_some_and(|e| e.telemetry.planet_mode));
            if is_macro_planet {
                self.macro_planet_mode = true;
                self.planet_mode = false;
                self.z16_active = false;
                let r_core = ent
                    .as_ref()
                    .map(|e| if e.telemetry.planet_r_core > 1.0 { e.telemetry.planet_r_core as f64 } else { 1280.0 })
                    .unwrap_or(1280.0);
                let spa = ent
                    .as_ref()
                    .map(|e| if e.telemetry.planet_shards_per_axis > 0 { e.telemetry.planet_shards_per_axis } else { st.planet_shards_per_axis })
                    .unwrap_or(st.planet_shards_per_axis);
                let seed = ent
                    .as_ref()
                    .map(|e| if e.telemetry.planet_seed != 0 { e.telemetry.planet_seed } else { st.planet_seed })
                    .unwrap_or(st.planet_seed);
                self.macro_planet_world.configure_if_needed(r_core, spa, gd.max(16), seed);
                let clim_u = st.take_macro_climate_updates();
                if !clim_u.is_empty() {
                    self.macro_planet_world.apply_climate_updates(clim_u);
                }
                let shard_u = st.take_macro_shard_brick_updates();
                if !shard_u.is_empty() {
                    self.macro_planet_world.apply_shard_updates(shard_u);
                }
            } else if is_planet {
                self.macro_planet_mode = false;
                self.planet_mode = true;
                self.z16_active = false;
                let r_core = ent
                    .as_ref()
                    .map(|e| if e.telemetry.planet_r_core > 1.0 { e.telemetry.planet_r_core } else { st.planet_r_core })
                    .unwrap_or(st.planet_r_core);
                self.planet_world.resize_if_needed(gw.max(16), gd.max(16), r_core.max(16.0));
                let updates = st.take_planet_brick_updates();
                if !updates.is_empty() {
                    self.planet_world.apply_updates(updates);
                }
                self.planet_world.rebuild_dirty_faces(3);
            } else {
                self.macro_planet_mode = false;
                self.planet_mode = false;
                self.world.resize_if_needed(gw, gh, gd);
                if gd >= 16 || st.has_z16_bricks {
                    self.z16_active = true;
                    self.z16_world.resize_if_needed(gw, gh, gd);
                    if !self.z16_cam_framed {
                        // Heroic auto-framing for Z=16..32 caves/bridges AND Z=64 3-strata Sky-Islands!
                        let (d0, p0, f0) = self.camera_preset.params_for_depth(true, gd);
                        self.camera.dist = if gd >= 48 { d0 } else { 36.0 };
                        self.camera.pitch = if gd >= 48 { p0 } else { 0.58 };
                        self.camera.fov_y = f0;
                        if gd >= 48 {
                            self.camera.target.y = gd as f32 * 0.45;
                        }
                        self.z16_cam_framed = true;
                    }
                    if gd > 32 {
                        let dirty_subs = st.take_dirty_z16_subbricks();
                        if !dirty_subs.is_empty() {
                            self.z16_world.apply_dirty_subbricks(
                                &st.z16_density_mat,
                                &st.z16_albedo_thermal,
                                &dirty_subs,
                            );
                        }
                    } else {
                        let dirty_z16 = st.take_dirty_z16_bricks();
                        if !dirty_z16.is_empty() {
                            self.z16_world.apply_dirty_bricks(
                                &st.z16_density_mat,
                                &st.z16_albedo_thermal,
                                &dirty_z16,
                            );
                        }
                    }
                } else {
                    self.z16_active = false;
                    let dirty = st.take_dirty_budget(20);
                    if !dirty.is_empty() {
                        self.world.update_dirty_chunks(&st.aplus_layers, &dirty);
                    }
                }
            }
            drop(st);
            (ent, rtt)
        } else {
            (None, 0.0)
        };

        // Wave 34B: Dedicated Live Dual-Scale Locally-Flat R=1,304m Macro Planet Interactive Frame Path
        if self.macro_planet_mode {
            let sun_d = entities
                .as_ref()
                .map(|e| {
                    let s = DVec3::new(
                        e.telemetry.sun_dir[0] as f64,
                        e.telemetry.sun_dir[1] as f64,
                        e.telemetry.sun_dir[2] as f64,
                    );
                    if s.length_squared() > 0.01 {
                        s.normalize()
                    } else {
                        DVec3::new(-0.72, 0.36, 0.59).normalize()
                    }
                })
                .unwrap_or_else(|| DVec3::new(-0.72, 0.36, 0.59).normalize());
            let sun_f = [sun_d.x as f32, sun_d.y as f32, sun_d.z as f32];
            let local_clim = entities
                .as_ref()
                .map(|e| [e.telemetry.local_wind_u, e.telemetry.local_wind_v, e.telemetry.local_cloud_cover])
                .unwrap_or([0.0, 0.0, 0.35]);

            let (hero_pos, hero_up, hero_tang_u, hero_tang_v, hero_fwd, physics_native_macro) = entities
                .as_ref()
                .and_then(|e| {
                    let pn = e.telemetry.physics_native;
                    e.players
                        .iter()
                        .find(|p| p.id == e.your_player_id)
                        .or_else(|| e.players.first())
                        .map(|me| (me, pn))
                })
                .map(|(me, pn)| {
                    let (pos, _, _, fwd) = self.macro_planet_world.entity_world_frame_f64(
                        me.face, me.x, me.y, me.z, me.aim_x, me.aim_y,
                    );
                    let n_virt = self.macro_planet_world.face_size_virtual();
                    let (u, v) = face_virtual_uv_f64(me.x as f64, me.y as f64, n_virt);
                    let (up, tu, tv) = macro_planet_world::face_tangent_basis_f64(me.face as usize, u, v);
                    (
                        DVec3::from_array(pos),
                        DVec3::from_array(up),
                        DVec3::from_array(tu),
                        DVec3::from_array(tv),
                        DVec3::from_array(fwd),
                        pn,
                    )
                })
                .unwrap_or_else(|| {
                    (
                        DVec3::new(0.0, 0.0, self.macro_planet_world.r_ocean()),
                        DVec3::Z,
                        DVec3::X,
                        DVec3::Y,
                        DVec3::X,
                        true,
                    )
                });

            self.macro_planet_camera.update_follow(
                hero_pos,
                hero_fwd,
                self.macro_planet_world.r_surf(),
                dt,
            );
            self.macro_planet_world
                .ensure_floating_origin(self.macro_planet_camera.eye_f64.to_array());
            self.macro_planet_world.rebuild_dirty(2);

            // Send 60 Hz camera-relative 6-DOF spherical WASD movement & live PhysicalActuator beams!
            let send_elapsed = self.last_input_send.elapsed();
            if self.input_dirty || send_elapsed >= Duration::from_millis(16) {
                self.last_input_send = Instant::now();
                self.input_dirty = false;
                if let Some(ref quic) = self.quic {
                    let mut f_in = 0.0f64;
                    let mut r_in = 0.0f64;
                    if !self.chat_open {
                        if self.keys.contains(&KeyCode::KeyW) { f_in += 1.0; }
                        if self.keys.contains(&KeyCode::KeyS) { f_in -= 1.0; }
                        if self.keys.contains(&KeyCode::KeyD) { r_in += 1.0; }
                        if self.keys.contains(&KeyCode::KeyA) { r_in -= 1.0; }
                    }
                    let (cam_fwd, cam_right) = self.macro_planet_camera.tangent_move_basis_f64(hero_up);
                    let move_3d = (cam_fwd * f_in + cam_right * r_in).clamp_length_max(1.0);
                    let mx = move_3d.dot(hero_tang_u) as f32;
                    let my = move_3d.dot(hero_tang_v) as f32;
                    let mlen = mx.hypot(my);
                    let (mx, my) = if mlen > 1.0 { (mx / mlen, my / mlen) } else { (mx, my) };

                    let aim_u = cam_fwd.dot(hero_tang_u) as f32;
                    let aim_v = cam_fwd.dot(hero_tang_v) as f32;
                    let alen = aim_u.hypot(aim_v);
                    let (aim_x, aim_y) = if alen > 0.01 {
                        (aim_u / alen, aim_v / alen)
                    } else if mlen > 0.05 {
                        (mx / mlen, my / mlen)
                    } else {
                        self.local_aim_dir
                    };
                    self.local_aim_dir = (aim_x, aim_y);

                    let shift_held = !self.chat_open
                        && (self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight));
                    let dash_req = !self.chat_open && (self.pending_dash || self.keys.contains(&KeyCode::Space));
                    let emit_matter = !self.chat_open
                        && (self.pending_ability_e
                            || self.keys.contains(&KeyCode::KeyE)
                            || (shift_held && self.mouse_left));
                    let emit_heat = !self.chat_open && self.mouse_left && !shift_held;
                    let emit_cryo = !self.chat_open && self.mouse_right;
                    let emit_pulse = !self.chat_open && self.pending_ult_r;

                    self.input_seq = self.input_seq.wrapping_add(1);
                    let now_ms = quic.state.lock().unwrap().client_time_ms();
                    let inp = ClientInputDatagram {
                        seq: self.input_seq,
                        move_x: mx,
                        move_y: my,
                        aim_x,
                        aim_y,
                        dash: dash_req,
                        melee: emit_cryo,
                        cast_bolt: emit_heat,
                        bolt_element: self.selected_element,
                        carve_ice: !self.chat_open && self.pending_carve,
                        cast_ability_e: !self.chat_open && self.pending_ability_e,
                        cast_ultimate_r: emit_pulse,
                        client_time_ms: now_ms,
                    };
                    quic.send_input(&inp);

                    if physics_native_macro && !self.chat_open {
                        let ready_cont = self.actuator_edge_fire
                            || self.last_actuator_send.elapsed() >= Duration::from_millis(85);
                        if emit_pulse || (ready_cont && (emit_heat || emit_cryo || emit_matter)) {
                            self.last_actuator_send = Instant::now();
                            self.actuator_edge_fire = false;
                            let mut thermal_flux = 0.0f32;
                            if emit_heat { thermal_flux += 1.0; }
                            if emit_cryo { thermal_flux -= 1.0; }
                            let mut sp_oil = 0.0f32;
                            let mut sp_water = 0.0f32;
                            let mut sp_smoke = 0.0f32;
                            if emit_matter {
                                match self.selected_element % 3 {
                                    0 => sp_oil = 1.0,
                                    1 => sp_water = 1.0,
                                    _ => sp_smoke = 1.0,
                                }
                            }
                            let mut radius = self.focus_radius;
                            let mut impulse = 0.0f32;
                            if emit_pulse {
                                thermal_flux = if shift_held { -1.0 } else { 1.0 };
                                radius = (self.focus_radius + 2.0).min(8.0);
                                impulse = 1.0;
                            }
                            quic.send_reliable_detached(ClientReliable::PhysicalActuator {
                                dx: aim_x * 12.0,
                                dy: aim_y * 12.0,
                                thermal_flux,
                                sp_water,
                                sp_oil,
                                sp_smoke,
                                focus_radius: radius,
                                impulse,
                            });
                        }
                    }

                    self.pending_dash = false;
                    self.pending_carve = false;
                    self.pending_ability_e = false;
                    self.pending_ult_r = false;
                }
            }

            let (Some(surface), Some(cfg), Some(renderer)) =
                (self.surface.as_ref(), self.surface_cfg.as_ref(), self.renderer.as_mut())
            else {
                return;
            };
            let out_w = cfg.width.max(1);
            let out_h = cfg.height.max(1);

            self.macro_planet_world.rebuild_dynamic_entities(entities.as_ref(), sim_time);
            renderer.upload_macro_planet_world(&mut self.macro_planet_world);
            if self.macro_planet_world.minimap_dirty {
                renderer.upload_minimap(
                    MACRO_MINIMAP_W as u32,
                    MACRO_MINIMAP_H as u32,
                    &self.macro_planet_world.minimap_rgba,
                );
                self.macro_planet_world.minimap_dirty = false;
            }

            let mmats = self.macro_planet_camera.compute_matrices(out_w as f32, out_h as f32);
            let hud_mats = mmats.to_hud_camera_matrices();
            let hud_batch = build_hud(
                out_w as f32,
                out_h as f32,
                &self.world,
                None,
                entities.as_ref(),
                &hud_mats,
                &HudConfig {
                    show_hud: self.show_hud,
                    show_telemetry: self.show_telemetry,
                    show_brain_inspector: self.brain_inspector_on,
                    brain_selected_idx: self.brain_selected_idx,
                    selected_element: self.selected_element,
                    focus_radius: self.focus_radius,
                    style_label: "MACRO-3D",
                    z16_active: false,
                    z16_style_label: "MACRO",
                    fps: self.fps_ema,
                    rtt_ms,
                    chat_open: self.chat_open,
                    chat_input: &self.chat_input,
                    chat_shout: self.chat_shout,
                    show_scoreboard: !self.chat_open && self.keys.contains(&KeyCode::Tab),
                    audio_muted: self.audio_muted,
                    audio_available: self.audio.is_some(),
                    cave_enclosure: 0.0,
                    show_squad_links: self.show_squad_links,
                    camera_preset_label: self.macro_planet_camera.mode.label(),
                    macro_eye_f64: Some(mmats.eye_f64),
                },
            );

            let surf_tex = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                _ => return,
            };
            let view = surf_tex.texture.create_view(&Default::default());
            renderer.render_macro_planet_into(
                &view,
                out_w,
                out_h,
                &mmats,
                sun_f,
                self.macro_planet_camera.exposure,
                sim_time,
                local_clim,
                &self.macro_planet_world,
                Some(&hud_batch),
            );
            renderer.queue.present(surf_tex);
            return;
        }

        // Wave 31B: Dedicated Live 6-Face Curved Spherical Planetary Interactive Frame Path
        if self.planet_mode {
            let sun_dir = entities
                .as_ref()
                .map(|e| {
                    let s = Vec3::from_array(e.telemetry.sun_dir);
                    if s.length_squared() > 0.01 {
                        s.normalize()
                    } else {
                        Vec3::new(-0.78, 0.28, 0.56).normalize()
                    }
                })
                .unwrap_or_else(|| Vec3::new(-0.78, 0.28, 0.56).normalize());

            let (hero_pos, hero_up, hero_tang_u, hero_tang_v, hero_fwd, physics_native_planet) = entities
                .as_ref()
                .and_then(|e| {
                    let pn = e.telemetry.physics_native;
                    e.players
                        .iter()
                        .find(|p| p.id == e.your_player_id)
                        .or_else(|| e.players.first())
                        .map(|me| (me, pn))
                })
                .map(|(me, pn)| {
                    let (pos, _, _, fwd) =
                        self.planet_world.entity_world_frame(me.face, me.x, me.y, me.z, me.aim_x, me.aim_y);
                    let nf = self.planet_world.face_size.max(1) as f32;
                    let u = (me.x / nf) * 2.0 - 1.0;
                    let v = (me.y / nf) * 2.0 - 1.0;
                    let (up, tu, tv) = planet_world::face_tangent_basis(me.face as usize, u, v);
                    (pos, up, tu, tv, fwd, pn)
                })
                .unwrap_or_else(|| {
                    (
                        Vec3::new(0.0, 0.0, self.planet_world.r_ocean()),
                        Vec3::Z,
                        Vec3::X,
                        Vec3::Y,
                        Vec3::X,
                        true,
                    )
                });

            self.planet_camera.update_follow(
                hero_pos,
                hero_fwd,
                sun_dir,
                self.planet_world.r_core,
                self.planet_world.depth as f32,
                dt,
            );

            // Send 60 Hz camera-relative 6-DOF spherical WASD movement & live PhysicalActuator beams!
            let send_elapsed = self.last_input_send.elapsed();
            if self.input_dirty || send_elapsed >= Duration::from_millis(16) {
                self.last_input_send = Instant::now();
                self.input_dirty = false;
                if let Some(ref quic) = self.quic {
                    let mut f_in = 0.0f32;
                    let mut r_in = 0.0f32;
                    if !self.chat_open {
                        if self.keys.contains(&KeyCode::KeyW) { f_in += 1.0; }
                        if self.keys.contains(&KeyCode::KeyS) { f_in -= 1.0; }
                        if self.keys.contains(&KeyCode::KeyD) { r_in += 1.0; }
                        if self.keys.contains(&KeyCode::KeyA) { r_in -= 1.0; }
                    }
                    let (cam_fwd, cam_right) = self.planet_camera.tangent_move_basis(hero_up);
                    let move_3d = (cam_fwd * f_in + cam_right * r_in).clamp_length_max(1.0);
                    let mx = move_3d.dot(hero_tang_u);
                    let my = move_3d.dot(hero_tang_v);
                    let mlen = mx.hypot(my);
                    let (mx, my) = if mlen > 1.0 { (mx / mlen, my / mlen) } else { (mx, my) };

                    // Aim straight along the camera's forward direction on the curved sphere!
                    let aim_u = cam_fwd.dot(hero_tang_u);
                    let aim_v = cam_fwd.dot(hero_tang_v);
                    let alen = aim_u.hypot(aim_v);
                    let (aim_x, aim_y) = if alen > 0.01 {
                        (aim_u / alen, aim_v / alen)
                    } else if mlen > 0.05 {
                        (mx / mlen, my / mlen)
                    } else {
                        self.local_aim_dir
                    };
                    self.local_aim_dir = (aim_x, aim_y);

                    let shift_held = !self.chat_open
                        && (self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight));
                    let dash_req = !self.chat_open && (self.pending_dash || self.keys.contains(&KeyCode::Space));
                    let emit_matter = !self.chat_open
                        && (self.pending_ability_e
                            || self.keys.contains(&KeyCode::KeyE)
                            || (shift_held && self.mouse_left));
                    let emit_heat = !self.chat_open && self.mouse_left && !shift_held;
                    let emit_cryo = !self.chat_open && self.mouse_right;
                    let emit_pulse = !self.chat_open && self.pending_ult_r;

                    self.input_seq = self.input_seq.wrapping_add(1);
                    let now_ms = quic.state.lock().unwrap().client_time_ms();
                    let inp = ClientInputDatagram {
                        seq: self.input_seq,
                        move_x: mx,
                        move_y: my,
                        aim_x,
                        aim_y,
                        dash: dash_req,
                        melee: emit_cryo,
                        cast_bolt: emit_heat,
                        bolt_element: self.selected_element,
                        carve_ice: !self.chat_open && self.pending_carve,
                        cast_ability_e: !self.chat_open && self.pending_ability_e,
                        cast_ultimate_r: emit_pulse,
                        client_time_ms: now_ms,
                    };
                    quic.send_input(&inp);

                    if physics_native_planet && !self.chat_open {
                        let ready_cont = self.actuator_edge_fire
                            || self.last_actuator_send.elapsed() >= Duration::from_millis(85);
                        if emit_pulse || (ready_cont && (emit_heat || emit_cryo || emit_matter)) {
                            self.last_actuator_send = Instant::now();
                            self.actuator_edge_fire = false;
                            let mut thermal_flux = 0.0f32;
                            if emit_heat { thermal_flux += 1.0; }
                            if emit_cryo { thermal_flux -= 1.0; }
                            let mut sp_oil = 0.0f32;
                            let mut sp_water = 0.0f32;
                            let mut sp_smoke = 0.0f32;
                            if emit_matter {
                                match self.selected_element % 3 {
                                    0 => sp_oil = 1.0,
                                    1 => sp_water = 1.0,
                                    _ => sp_smoke = 1.0,
                                }
                            }
                            let mut radius = self.focus_radius;
                            let mut impulse = 0.0f32;
                            if emit_pulse {
                                thermal_flux = if shift_held { -1.0 } else { 1.0 };
                                radius = (self.focus_radius + 2.0).min(8.0);
                                impulse = 1.0;
                            }
                            quic.send_reliable_detached(ClientReliable::PhysicalActuator {
                                dx: aim_x * 9.0,
                                dy: aim_y * 9.0,
                                thermal_flux,
                                sp_water,
                                sp_oil,
                                sp_smoke,
                                focus_radius: radius,
                                impulse,
                            });
                        }
                    }

                    self.pending_dash = false;
                    self.pending_carve = false;
                    self.pending_ability_e = false;
                    self.pending_ult_r = false;
                }
            }

            let (Some(surface), Some(cfg), Some(renderer)) =
                (self.surface.as_ref(), self.surface_cfg.as_ref(), self.renderer.as_mut())
            else {
                return;
            };
            let out_w = cfg.width.max(1);
            let out_h = cfg.height.max(1);

            self.planet_world.rebuild_dynamic_entities(entities.as_ref(), sim_time);
            renderer.upload_planet_world(&mut self.planet_world);
            if self.planet_world.minimap_dirty {
                renderer.upload_minimap(
                    PLANET_MINIMAP_W as u32,
                    PLANET_MINIMAP_H as u32,
                    &self.planet_world.minimap_rgba,
                );
                self.planet_world.minimap_dirty = false;
            }

            let pmats = self.planet_camera.compute_matrices(out_w as f32, out_h as f32);
            let hud_mats = pmats.to_camera_matrices();
            let hud_batch = build_hud(
                out_w as f32,
                out_h as f32,
                &self.world,
                None,
                entities.as_ref(),
                &hud_mats,
                &HudConfig {
                    show_hud: self.show_hud,
                    show_telemetry: self.show_telemetry,
                    show_brain_inspector: self.brain_inspector_on,
                    brain_selected_idx: self.brain_selected_idx,
                    selected_element: self.selected_element,
                    focus_radius: self.focus_radius,
                    style_label: "PLANET-3D",
                    z16_active: false,
                    z16_style_label: "PLANET",
                    fps: self.fps_ema,
                    rtt_ms,
                    chat_open: self.chat_open,
                    chat_input: &self.chat_input,
                    chat_shout: self.chat_shout,
                    show_scoreboard: !self.chat_open && self.keys.contains(&KeyCode::Tab),
                    audio_muted: self.audio_muted,
                    audio_available: self.audio.is_some(),
                    cave_enclosure: 0.0,
                    show_squad_links: self.show_squad_links,
                    camera_preset_label: self.planet_camera.mode.label(),
                    macro_eye_f64: None,
                },
            );

            let surf_tex = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                _ => return,
            };
            let view = surf_tex.texture.create_view(&Default::default());
            renderer.render_planet_into(
                &view,
                out_w,
                out_h,
                &pmats,
                sun_dir,
                self.planet_camera.exposure,
                sim_time,
                &self.planet_world,
                Some(&hud_batch),
            );
            renderer.queue.present(surf_tex);
            return;
        }

        // Compute camera-basis WASD right away so client-side hero prediction runs every single frame at native FPS!
        let pre_fwd_xz = Vec3::new(-self.camera.yaw.sin(), 0.0, -self.camera.yaw.cos()).normalize_or(Vec3::NEG_Z);
        let pre_right_xz = Vec3::new(-pre_fwd_xz.z, 0.0, pre_fwd_xz.x);
        let (mut f_in_now, mut r_in_now) = (0.0f32, 0.0f32);
        if !self.chat_open {
            if self.keys.contains(&KeyCode::KeyW) { f_in_now += 1.0; }
            if self.keys.contains(&KeyCode::KeyS) { f_in_now -= 1.0; }
            if self.keys.contains(&KeyCode::KeyD) { r_in_now += 1.0; }
            if self.keys.contains(&KeyCode::KeyA) { r_in_now -= 1.0; }
        }
        let local_move_v = (pre_fwd_xz * f_in_now + pre_right_xz * r_in_now).clamp_length_max(1.0);

        // Exact server-matched client-side hero prediction + `ack_input_seq` replay reconciliation
        // + smooth velocity-extrapolated interpolation for all remote entities/projectiles/rafts!
        let mut new_snap_arrived = false;
        if let Some(ref mut ent) = entities {
            if ent.sim_tick != self.last_snap_tick {
                self.last_snap_tick = ent.sim_tick;
                self.snap_age_s = 0.0;
                new_snap_arrived = true;
            } else {
                self.snap_age_s = (self.snap_age_s + dt).min(0.16);
            }
            let alpha_other = 1.0 - (-18.0 * dt).exp();
            let extrap_t = self.snap_age_s.min(0.10);
            let gw_max = (self.world.grid_w.saturating_sub(2)) as f32;
            let gh_max = (self.world.grid_h.saturating_sub(2)) as f32;
            let bodies_ref = ent.bodies.clone();

            let mut live_pids = HashSet::with_capacity(ent.players.len());
            for p in &mut ent.players {
                live_pids.insert(p.id);
                if p.id == ent.your_player_id {
                    let hasted = (p.status_flags & STATUS_HASTE) != 0;
                    let pred_dashing = p.dashing
                        || self.hero_dash_pred_until.map_or(false, |t| Instant::now() < t);

                    // 1. On each authoritative server snapshot, drop acknowledged inputs & replay unacked ones
                    if new_snap_arrived || self.pred_hero_pos.is_none() {
                        while let Some(front) = self.pending_inputs.front() {
                            // Wrapping-safe ack check
                            if p.ack_input_seq.wrapping_sub(front.seq) < 10_000 {
                                self.pending_inputs.pop_front();
                            } else {
                                break;
                            }
                        }
                        let (mut rx, mut ry, mut rz, mut rvz) = (p.x, p.y, p.z, p.vz);
                        for pin in &self.pending_inputs {
                            if self.z16_active {
                                let mlen = pin.move_x.hypot(pin.move_y);
                                let (dx, dy) = if mlen >= 0.01 { (pin.move_x / mlen, pin.move_y / mlen) } else { (0.0, 0.0) };
                                let speed = if pin.dash {
                                    42.0
                                } else if mlen < 0.01 {
                                    0.0
                                } else {
                                    let mut s = hero_class_base_speed(p.class_id);
                                    if p.carrying_relic { s *= 0.78; }
                                    if hasted { s *= 1.32; }
                                    s
                                };
                                if pin.dash && rvz < 0.5 {
                                    rvz = 6.2;
                                }
                                let ((nx, ny, nz), (_, _, nvz), _) = self.z16_world.step_capsule_3d(
                                    (rx, ry, rz),
                                    (dx * speed, dy * speed, rvz),
                                    0.55,
                                    1.80,
                                    pin.dt,
                                );
                                rx = nx;
                                ry = ny;
                                rz = nz;
                                rvz = nvz;
                            } else {
                                let (nx, ny) = predict_hero_step(
                                    &self.world,
                                    rx,
                                    ry,
                                    p.z,
                                    pin.move_x,
                                    pin.move_y,
                                    pin.dt,
                                    p.class_id,
                                    p.carrying_relic,
                                    pin.dash,
                                    hasted,
                                    &bodies_ref,
                                );
                                rx = nx;
                                ry = ny;
                            }
                        }
                        if let Some((old_x, old_y, old_z)) = self.pred_hero_pos {
                            let old_disp_x = old_x + self.hero_visual_err.0;
                            let old_disp_y = old_y + self.hero_visual_err.1;
                            let err = (old_disp_x - rx).hypot(old_disp_y - ry);
                            if err > 10.0 || p.hp <= 0 {
                                // Teleport / respawn -> snap cleanly
                                self.hero_visual_err = (0.0, 0.0);
                            } else if err > 0.06 {
                                // Absorb tiny residual into visual offset so hero NEVER snaps backwards!
                                self.hero_visual_err = (old_disp_x - rx, old_disp_y - ry);
                            }
                            let next_z = if self.z16_active {
                                old_z + (rz - old_z) * alpha_other
                            } else {
                                old_z + (p.z - old_z) * alpha_other
                            };
                            self.pred_hero_pos = Some((rx, ry, next_z));
                            self.pred_hero_vz = rvz;
                        } else {
                            self.pred_hero_pos = Some((rx, ry, if self.z16_active { rz } else { p.z }));
                            self.pred_hero_vz = rvz;
                            self.hero_visual_err = (0.0, 0.0);
                        }
                    }

                    // 2. Advance local predicted hero smoothly every frame at native FPS
                    let (mut px, mut py, mut pz) = self.pred_hero_pos.unwrap_or((p.x, p.y, p.z));
                    if p.hp > 0 {
                        let (step_mx, step_my) = if pred_dashing && local_move_v.length_squared() < 0.01 {
                            self.local_aim_dir
                        } else {
                            (local_move_v.x, local_move_v.z)
                        };
                        if self.z16_active {
                            let mlen = step_mx.hypot(step_my);
                            let (dx, dy) = if mlen >= 0.01 { (step_mx / mlen, step_my / mlen) } else { (0.0, 0.0) };
                            let speed = if pred_dashing {
                                42.0
                            } else if mlen < 0.01 {
                                0.0
                            } else {
                                let mut s = hero_class_base_speed(p.class_id);
                                if p.carrying_relic { s *= 0.78; }
                                if hasted { s *= 1.32; }
                                s
                            };
                            let ((nx, ny, nz), (_, _, nvz), _) = self.z16_world.step_capsule_3d(
                                (px, py, pz),
                                (dx * speed, dy * speed, self.pred_hero_vz),
                                0.55,
                                1.80,
                                dt,
                            );
                            px = nx;
                            py = ny;
                            pz = nz;
                            self.pred_hero_vz = nvz;
                        } else {
                            let (nx, ny) = predict_hero_step(
                                &self.world,
                                px,
                                py,
                                pz,
                                step_mx,
                                step_my,
                                dt,
                                p.class_id,
                                p.carrying_relic,
                                pred_dashing,
                                hasted,
                                &bodies_ref,
                            );
                            px = nx;
                            py = ny;
                            pz += (p.z - pz) * alpha_other;
                        }
                    } else {
                        pz += (p.z - pz) * alpha_other;
                    }
                    self.pred_hero_pos = Some((px, py, pz));

                    // Smoothly decay any tiny visual reconciliation residual
                    let decay = (-16.0 * dt).exp();
                    self.hero_visual_err.0 *= decay;
                    self.hero_visual_err.1 *= decay;
                    if self.hero_visual_err.0.abs() < 0.005 { self.hero_visual_err.0 = 0.0; }
                    if self.hero_visual_err.1.abs() < 0.005 { self.hero_visual_err.1 = 0.0; }

                    p.x = (px + self.hero_visual_err.0).clamp(1.0, gw_max);
                    p.y = (py + self.hero_visual_err.1).clamp(1.0, gh_max);
                    p.z = pz;
                    if local_move_v.length_squared() > 0.01 && p.hp > 0 {
                        let sp = hero_class_base_speed(p.class_id);
                        p.vx = local_move_v.x * sp;
                        p.vy = local_move_v.z * sp;
                    }
                    if pred_dashing {
                        p.dashing = true;
                    }
                } else {
                    let target_x = (p.x + p.vx * extrap_t).clamp(1.0, gw_max);
                    let target_y = (p.y + p.vy * extrap_t).clamp(1.0, gh_max);
                    let target_z = (p.z + p.vz * extrap_t).clamp(0.0, (self.world.grid_d.saturating_sub(1)) as f32);
                    let entry = self.smooth_players.entry(p.id).or_insert((p.x, p.y, p.z));
                    let err = (entry.0 - p.x).hypot(entry.1 - p.y);
                    if err > 14.0 {
                        *entry = (p.x, p.y, p.z);
                    } else {
                        entry.0 += (target_x - entry.0) * alpha_other;
                        entry.1 += (target_y - entry.1) * alpha_other;
                        entry.2 += (target_z - entry.2) * alpha_other;
                    }
                    p.x = entry.0;
                    p.y = entry.1;
                    p.z = entry.2;
                }
            }
            self.smooth_players.retain(|id, _| live_pids.contains(id));

            // Inject short-lived local predicted actuator pulses for 0 ms visual click responsiveness
            let now_inst = Instant::now();
            self.local_actuator_pulses.retain(|(exp, _)| *exp > now_inst);
            for (_, lp) in &self.local_actuator_pulses {
                ent.actuator_pulses.push(lp.clone());
            }

            let mut live_prids = HashSet::with_capacity(ent.projectiles.len());
            for pr in &mut ent.projectiles {
                live_prids.insert(pr.id);
                let tx = (pr.x + pr.vx * extrap_t).clamp(0.0, gw_max);
                let ty = (pr.y + pr.vy * extrap_t).clamp(0.0, gh_max);
                let tz = pr.z + pr.vz * extrap_t;
                let entry = self.smooth_projs.entry(pr.id).or_insert((pr.x, pr.y, pr.z));
                if (entry.0 - pr.x).hypot(entry.1 - pr.y) > 16.0 {
                    *entry = (pr.x, pr.y, pr.z);
                } else {
                    let a = 1.0 - (-24.0 * dt).exp();
                    entry.0 += (tx - entry.0) * a;
                    entry.1 += (ty - entry.1) * a;
                    entry.2 += (tz - entry.2) * a;
                }
                pr.x = entry.0;
                pr.y = entry.1;
                pr.z = entry.2;
            }
            self.smooth_projs.retain(|id, _| live_prids.contains(id));

            let mut live_bids = HashSet::with_capacity(ent.bodies.len());
            for cb in &mut ent.bodies {
                live_bids.insert(cb.id);
                let entry = self.smooth_bodies.entry(cb.id).or_insert((cb.x, cb.y, cb.z));
                entry.0 += (cb.x - entry.0) * alpha_other;
                entry.1 += (cb.y - entry.1) * alpha_other;
                entry.2 += (cb.z - entry.2) * alpha_other;
                cb.x = entry.0;
                cb.y = entry.1;
                cb.z = entry.2;
            }
            self.smooth_bodies.retain(|id, _| live_bids.contains(id));
        }

        // Smooth 3D camera follow onto your predicted hero's exact altitude
        if let Some(ref ent) = entities {
            if let Some(me) = ent.players.iter().find(|p| p.id == ent.your_player_id) {
                let (wx, wz) = if self.z16_active {
                    self.z16_world.grid_to_world_xz(me.x, me.y)
                } else {
                    self.world.grid_to_world_xz(me.x, me.y)
                };
                let gy = if self.z16_active {
                    me.z
                } else {
                    self.world.sample_3d_y(me.x, me.y, me.z)
                };
                let desired = if self.z16_active {
                    Vec3::new(wx, gy + 1.0, wz)
                } else {
                    Vec3::new(wx, gy + 0.8, wz)
                };
                let cam_alpha = 1.0 - (-12.0 * dt).exp();
                self.camera.target = self.camera.target.lerp(desired, cam_alpha);
            }
        }

        let (Some(surface), Some(cfg), Some(renderer)) =
            (self.surface.as_ref(), self.surface_cfg.as_ref(), self.renderer.as_mut())
        else {
            return;
        };

        renderer.set_z16_mode(self.z16_active, self.z16_style_mode);
        if self.z16_active {
            renderer.upload_z16_world(&mut self.z16_world);
            if self.z16_world.minimap_dirty {
                renderer.upload_minimap(
                    self.z16_world.width as u32,
                    self.z16_world.height as u32,
                    &self.z16_world.minimap_rgba,
                );
                self.z16_world.minimap_dirty = false;
            }
        } else if self.world.minimap_dirty {
            renderer.upload_minimap(
                self.world.grid_w as u32,
                self.world.grid_h as u32,
                &self.world.minimap_rgba,
            );
            self.world.minimap_dirty = false;
        }

        let out_w = cfg.width.max(1);
        let out_h = cfg.height.max(1);
        let atmo = entities
            .as_ref()
            .map(|e| Atmosphere {
                weather_kind: e.telemetry.weather_kind,
                day_time: e.telemetry.day_time,
                wind_x: e.telemetry.wind_x,
                wind_y: e.telemetry.wind_y,
            })
            .unwrap_or_default();
        let sun_dir = Approach4Renderer::sun_dir(atmo);
        let mats = self.camera.compute_matrices(out_w as f32, out_h as f32, sun_dir);

        // Sample local hero 3D position for Cave Acoustic Enclosure (Schroeder Reverb + HUD)
        // AND Smart 3D Cave & Under-Bridge Ceiling Cutaway!
        let (hero_gx, hero_gy, hero_gz, hero_team, hero_stamina, hero_hp) = entities
            .as_ref()
            .and_then(|e| e.players.iter().find(|p| p.id == e.your_player_id))
            .map(|me| (me.x, me.y, me.z, me.team, me.stamina, me.hp))
            .unwrap_or((self.world.grid_w as f32 * 0.5, self.world.grid_h as f32 * 0.5, 1.0, 0, 100, 100));

        if self.z16_active {
            let (hwx, hwz) = self.z16_world.grid_to_world_xz(hero_gx, hero_gy);
            let hero_world_pos = Vec3::new(hwx, hero_gz, hwz);
            let cam_world_pos = mats.origin + mats.rel_eye;

            let raw_enc = self.z16_world.sample_cave_enclosure(hero_gx, hero_gy, hero_gz);
            let enc_alpha = 1.0 - (-8.0 * dt).exp();
            self.cave_enclosure_ema += (raw_enc - self.cave_enclosure_ema) * enc_alpha;
            if self.cave_enclosure_ema < 0.005 {
                self.cave_enclosure_ema = 0.0;
            }

            let want_cutaway = self
                .z16_world
                .should_cutaway_ceiling(hero_gx, hero_gy, hero_gz, cam_world_pos);
            let cut_tgt = if want_cutaway { 1.0 } else { 0.0 };
            let cut_alpha = 1.0 - (-10.0 * dt).exp();
            self.cutaway_weight += (cut_tgt - self.cutaway_weight) * cut_alpha;
            if self.cutaway_weight < 0.005 {
                self.cutaway_weight = 0.0;
            }
            renderer.set_z16_hero_cutaway(hero_world_pos, self.cutaway_weight);
        } else {
            self.cave_enclosure_ema = 0.0;
            self.cutaway_weight = 0.0;
            renderer.set_z16_hero_cutaway(Vec3::ZERO, 0.0);
        }

        // Update Real-Time 3D Spatial Audio Engine (ambient waterfalls/wildfire/wind + cave reverb + spatial SFX)
        if let Some(ref audio) = self.audio {
            audio.set_cave_enclosure(self.cave_enclosure_ema);
            let fwd_xz = Vec3::new(mats.fwd.x, 0.0, mats.fwd.z).normalize_or(Vec3::NEG_Z);
            let right_xz = Vec3::new(-fwd_xz.z, 0.0, fwd_xz.x);
            let listener_pos = self.camera.target;

            let spatial_pan_gain = |src: Vec3, ref_dist: f32| -> (f32, f32) {
                let d = src - listener_pos;
                let dist = d.length();
                let pan = Vec3::new(d.x, 0.0, d.z)
                    .normalize_or(Vec3::ZERO)
                    .dot(right_xz)
                    .clamp(-1.0, 1.0);
                let gain = (1.0 / (1.0 + (dist / ref_dist).powi(2))).clamp(0.0, 1.0);
                (pan, gain)
            };

            let (nearest_wf, nearby_fire) = if self.z16_active {
                self.z16_world.sample_ambient_audio_sources(listener_pos)
            } else {
                self.world.sample_ambient_audio_sources(listener_pos)
            };
            let (wf_gain, wf_pan) = if let Some((wp, _)) = nearest_wf {
                let (p, g) = spatial_pan_gain(wp, 38.0);
                (g, p)
            } else {
                (0.0, 0.0)
            };
            let (fire_gain, fire_pan) = if let Some((fp, inten)) = nearby_fire {
                let (p, g) = spatial_pan_gain(fp, 34.0);
                ((g * (inten / 16.0).clamp(0.20, 1.0)).clamp(0.0, 1.0), p)
            } else {
                (0.0, 0.0)
            };
            let wind_mag = atmo.wind_x.hypot(atmo.wind_y);
            audio.update_ambient(wf_gain, wf_pan, fire_gain, fire_pan, atmo.weather_kind, wind_mag);

            if let Some(ref ent) = entities {
                if ent.sim_tick != self.last_audio_sim_tick {
                    if ent.sim_tick < self.last_audio_sim_tick {
                        self.seen_proj_ids.clear();
                        self.seen_event_ids.clear();
                        self.seen_bubble_ticks.clear();
                    }
                    self.last_audio_sim_tick = ent.sim_tick;

                    // New elemental projectile casts
                    let mut active_projs = HashSet::with_capacity(ent.projectiles.len());
                    for pr in &ent.projectiles {
                        active_projs.insert(pr.id);
                        if !self.seen_proj_ids.contains(&pr.id) {
                            let (wx, wz) = self.world.grid_to_world_xz(pr.x, pr.y);
                            let wy = self.world.sample_3d_y(pr.x, pr.y, pr.z);
                            let (pan, gain) = spatial_pan_gain(Vec3::new(wx, wy, wz), 32.0);
                            let kind = match pr.element % 3 {
                                0 => SfxKind::FireCast,
                                1 => SfxKind::FrostCast,
                                _ => SfxKind::OilCast,
                            };
                            audio.trigger_sfx(kind, (pr.id & 0xFF) as u8, gain * 0.72, pan);
                        }
                    }
                    self.seen_proj_ids = active_projs;

                    // New combat / combo / kill / shrine / relic events
                    for ev in &ent.events {
                        if !self.seen_event_ids.contains(&ev.id) {
                            self.seen_event_ids.push_back(ev.id);
                            while self.seen_event_ids.len() > 64 {
                                self.seen_event_ids.pop_front();
                            }
                            let (wx, wz) = self.world.grid_to_world_xz(ev.x, ev.y);
                            let wy = self.world.sample_ground_y(ev.x, ev.y);
                            let (pan, gain) = spatial_pan_gain(Vec3::new(wx, wy, wz), 46.0);
                            match ev.kind {
                                1 => audio.trigger_sfx(SfxKind::KillChord, 0, gain.max(0.32) * 0.85, pan),
                                2 => audio.trigger_sfx(SfxKind::SteamCombo, (ev.id & 0xFF) as u8, gain.max(0.22) * 0.88, pan),
                                3 | 4 => audio.trigger_sfx(SfxKind::ShrineOrRelic, 0, gain.max(0.45) * 0.80, pan),
                                _ => {}
                            }
                        }
                    }

                    // New 3D Acoustic Speech Bubbles -> SNR-Filtered Neural-Voice Synth Chirps!
                    for sb in &ent.speech_bubbles {
                        if self.seen_bubble_ticks.get(&sb.speaker_id).copied() != Some(sb.tick) {
                            self.seen_bubble_ticks.insert(sb.speaker_id, sb.tick);
                            let (wx, wz) = self.world.grid_to_world_xz(sb.x, sb.y);
                            let wy = self.world.sample_3d_y(sb.x, sb.y, sb.z);
                            let (pan, dist_g) = spatial_pan_gain(Vec3::new(wx, wy, wz), 52.0);
                            let gain = dist_g.max(0.36) * 0.90;
                            let sym = sb.symbol_id.wrapping_add((sb.speaker_id & 0x0F) as u8);
                            audio.trigger_voice_chirp(sym, sb.snr_db, gain, pan);
                        }
                    }
                }
            }
        }

        // Compute exact every-frame mouse-ray cursor aiming on the hero's actual 3D altitude plane!
        let fwd_xz = Vec3::new(mats.fwd.x, 0.0, mats.fwd.z).normalize_or(Vec3::NEG_Z);
        let right_xz = Vec3::new(-fwd_xz.z, 0.0, fwd_xz.x);
        let hero_plane_y = if self.z16_active {
            hero_gz
        } else {
            self.world.sample_3d_y(hero_gx, hero_gy, hero_gz)
        };

        let mut aim_x = fwd_xz.x;
        let mut aim_y = fwd_xz.z;
        let mut cursor_grid: Option<(f32, f32)> = None;
        if let Some((hit_wx, hit_wz)) = self.camera.screen_to_world_xz(
            &mats,
            self.mouse_pos.0,
            self.mouse_pos.1,
            out_w as f32,
            out_h as f32,
            hero_plane_y,
        ) {
            let (hgx, hgy) = self.world.world_xz_to_grid(hit_wx, hit_wz);
            cursor_grid = Some((hgx, hgy));
            self.last_cursor_grid = cursor_grid;
            let dx = hgx - hero_gx;
            let dy = hgy - hero_gy;
            if dx.hypot(dy) > 0.05 {
                aim_x = dx;
                aim_y = dy;
            }
        }
        let alen = aim_x.hypot(aim_y);
        if alen > 0.01 {
            self.local_aim_dir = (aim_x / alen, aim_y / alen);
            // Zero-latency 120 FPS client-side hero facing toward exact mouse cursor!
            if let Some(ref mut ent) = entities {
                if let Some(me) = ent.players.iter_mut().find(|p| p.id == ent.your_player_id) {
                    me.aim_x = self.local_aim_dir.0;
                    me.aim_y = self.local_aim_dir.1;
                }
            }
        }

        // Send 60 Hz (or immediate edge-triggered 0 ms!) inputs + physical actuators to server
        let send_elapsed = self.last_input_send.elapsed();
        let should_send_input = self.input_dirty || send_elapsed >= Duration::from_millis(16);
        if should_send_input {
            let step_dt = send_elapsed.as_secs_f32().clamp(0.004, 0.05);
            self.last_input_send = Instant::now();
            self.input_dirty = false;
            if let Some(ref quic) = self.quic {
                let mut f_in = 0.0f32;
                let mut r_in = 0.0f32;
                if !self.chat_open {
                    if self.keys.contains(&KeyCode::KeyW) { f_in += 1.0; }
                    if self.keys.contains(&KeyCode::KeyS) { f_in -= 1.0; }
                    if self.keys.contains(&KeyCode::KeyD) { r_in += 1.0; }
                    if self.keys.contains(&KeyCode::KeyA) { r_in -= 1.0; }
                }
                let move_v = (fwd_xz * f_in + right_xz * r_in).clamp_length_max(1.0);

                let physics_native = entities
                    .as_ref()
                    .map(|e| e.telemetry.physics_native)
                    .unwrap_or(false);

                self.input_seq = self.input_seq.wrapping_add(1);
                let now_ms = quic.state.lock().unwrap().client_time_ms();
                let shift_held = !self.chat_open
                    && (self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight));

                let dash_req = !self.chat_open && (self.pending_dash || self.keys.contains(&KeyCode::Space));
                if dash_req && hero_stamina >= 25 && hero_hp > 0
                    && self.hero_dash_pred_until.map_or(true, |t| Instant::now() >= t)
                {
                    self.hero_dash_pred_until = Some(Instant::now() + Duration::from_millis(200));
                }

                // Record unacknowledged movement input for exact `ack_input_seq` replay reconciliation
                self.pending_inputs.push_back(PendingMoveInput {
                    seq: self.input_seq,
                    move_x: move_v.x,
                    move_y: move_v.z,
                    dt: step_dt,
                    dash: dash_req,
                });
                while self.pending_inputs.len() > 90 {
                    self.pending_inputs.pop_front();
                }

                if physics_native && !self.chat_open {
                    let emit_matter = self.pending_ability_e
                        || self.keys.contains(&KeyCode::KeyE)
                        || (shift_held && self.mouse_left);
                    let emit_heat = self.mouse_left && !shift_held;
                    let emit_cryo = self.mouse_right;
                    let emit_pulse = self.pending_ult_r;

                    let ready_cont = self.actuator_edge_fire
                        || self.last_actuator_send.elapsed() >= Duration::from_millis(85);
                    if emit_pulse || (ready_cont && (emit_heat || emit_cryo || emit_matter)) {
                        self.last_actuator_send = Instant::now();
                        self.actuator_edge_fire = false;
                        let mut thermal_flux = 0.0f32;
                        if emit_heat { thermal_flux += 1.0; }
                        if emit_cryo { thermal_flux -= 1.0; }
                        let mut sp_oil = 0.0f32;
                        let mut sp_water = 0.0f32;
                        let mut sp_smoke = 0.0f32;
                        if emit_matter {
                            match self.selected_element % 3 {
                                0 => sp_oil = 1.0,
                                1 => sp_water = 1.0,
                                _ => sp_smoke = 1.0,
                            }
                        }
                        let mut radius = self.focus_radius;
                        let mut impulse = 0.0f32;
                        if emit_pulse {
                            thermal_flux = if shift_held { -1.0 } else { 1.0 };
                            radius = (self.focus_radius + 2.0).min(8.0);
                            impulse = 1.0;
                        }
                        let adist = aim_x.hypot(aim_y);
                        let (dx, dy) = if adist > 22.0 {
                            (aim_x * (22.0 / adist), aim_y * (22.0 / adist))
                        } else if adist < 1.5 {
                            (fwd_xz.x * 8.0, fwd_xz.z * 8.0)
                        } else {
                            (aim_x, aim_y)
                        };
                        quic.send_reliable_detached(ClientReliable::PhysicalActuator {
                            dx,
                            dy,
                            thermal_flux,
                            sp_water,
                            sp_oil,
                            sp_smoke,
                            focus_radius: radius,
                            impulse,
                        });
                        // Instant 0 ms local predicted volumetric actuator beam + crisp audio response!
                        let tx = (hero_gx + dx).clamp(2.0, (self.world.grid_w.saturating_sub(3)) as f32);
                        let ty = (hero_gy + dy).clamp(2.0, (self.world.grid_h.saturating_sub(3)) as f32);
                        let tz1 = if self.z16_active {
                            self.z16_world.sample_surface_y(tx, ty)
                        } else {
                            hero_gz
                        };
                        let local_pulse = ActuatorPulseSnap {
                            x0: hero_gx,
                            y0: hero_gy,
                            z0: hero_gz,
                            x1: tx,
                            y1: ty,
                            z1: tz1,
                            thermal_flux,
                            sp_water,
                            sp_oil,
                            sp_smoke,
                            focus_radius: radius,
                            impulse,
                            team: hero_team,
                            face: 0,
                        };
                        self.local_actuator_pulses.push((Instant::now() + Duration::from_millis(160), local_pulse.clone()));
                        if let Some(ref mut ent) = entities {
                            ent.actuator_pulses.push(local_pulse);
                        }
                        if let Some(ref audio) = self.audio {
                            let sfx = if thermal_flux > 0.2 {
                                SfxKind::FireCast
                            } else if thermal_flux < -0.2 {
                                SfxKind::FrostCast
                            } else {
                                SfxKind::OilCast
                            };
                            audio.trigger_sfx(sfx, (self.input_seq & 0xFF) as u8, 0.52, 0.0);
                        }
                    }
                }

                let inp = ClientInputDatagram {
                    seq: self.input_seq,
                    move_x: move_v.x,
                    move_y: move_v.z,
                    aim_x,
                    aim_y,
                    dash: dash_req,
                    melee: !self.chat_open && !physics_native && self.mouse_right,
                    cast_bolt: !self.chat_open && !physics_native && self.mouse_left,
                    bolt_element: self.selected_element,
                    carve_ice: !self.chat_open && self.pending_carve,
                    cast_ability_e: !self.chat_open && !physics_native && self.pending_ability_e,
                    cast_ultimate_r: !self.chat_open && !physics_native && self.pending_ult_r,
                    client_time_ms: now_ms,
                };
                self.pending_dash = false;
                self.pending_carve = false;
                self.pending_ability_e = false;
                self.pending_ult_r = false;
                quic.send_input(&inp);
            }
        }

        let aim_reticle = cursor_grid.map(|(cgx, cgy)| {
            let shift_held = !self.chat_open
                && (self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight));
            let col = if self.mouse_right {
                [0.42, 0.92, 1.0]
            } else if shift_held || self.keys.contains(&KeyCode::KeyE) {
                match self.selected_element % 3 {
                    0 => [0.76, 0.36, 0.98],
                    1 => [0.34, 0.84, 1.0],
                    _ => [0.68, 0.72, 0.84],
                }
            } else if self.mouse_left {
                [1.0, 0.56, 0.16]
            } else {
                [0.44, 0.98, 0.68]
            };
            AimReticle {
                gx: cgx.clamp(2.0, (self.world.grid_w.saturating_sub(3)) as f32),
                gy: cgy.clamp(2.0, (self.world.grid_h.saturating_sub(3)) as f32),
                gz: hero_gz,
                radius: self.focus_radius,
                color: col,
            }
        });

        let style = if self.style_clean {
            RenderStyle::clean_3d()
        } else {
            RenderStyle::pixel_art(if out_h >= 2100 { 3 } else { 2 })
        };
        let z16_ref = if self.z16_active { Some(&self.z16_world) } else { None };
        let frame = assemble_live_frame(
            &self.world,
            z16_ref,
            entities.as_ref(),
            &self.surfels,
            &mats,
            self.camera.target,
            self.camera.dist,
            sim_time,
            self.layer_slice_mode,
            aim_reticle,
            self.show_squad_links,
        );
        let hud_batch = build_hud(
            out_w as f32,
            out_h as f32,
            &self.world,
            z16_ref,
            entities.as_ref(),
            &mats,
            &HudConfig {
                show_hud: self.show_hud,
                show_telemetry: self.show_telemetry,
                show_brain_inspector: self.brain_inspector_on,
                brain_selected_idx: self.brain_selected_idx,
                selected_element: self.selected_element,
                focus_radius: self.focus_radius,
                style_label: style.label,
                z16_active: self.z16_active,
                z16_style_label: z16_style_label(self.z16_style_mode),
                fps: self.fps_ema,
                rtt_ms,
                chat_open: self.chat_open,
                chat_input: &self.chat_input,
                chat_shout: self.chat_shout,
                show_scoreboard: !self.chat_open && self.keys.contains(&KeyCode::Tab),
                audio_muted: self.audio_muted,
                audio_available: self.audio.is_some(),
                cave_enclosure: self.cave_enclosure_ema,
                show_squad_links: self.show_squad_links,
                camera_preset_label: self.camera_preset.label(),
                macro_eye_f64: None,
            },
        );

        let surf_tex = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = surf_tex.texture.create_view(&Default::default());
        renderer.render_into(
            &view,
            out_w,
            out_h,
            &style,
            atmo,
            &self.camera,
            &mats,
            sim_time,
            &frame,
            Some(&hud_batch),
        );
        renderer.queue.present(surf_tex);
    }
}

impl ApplicationHandler for InteractiveApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Frostmarsh Unified v2 — Shattered Reach (Live Neural World)")
            .with_inner_size(LogicalSize::new(1440.0, 900.0));
        let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let sz = win.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(win.clone()).expect("create surface");
        let adapter = self
            .rt
            .block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            }))
            .expect("request adapter");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let renderer = self
            .rt
            .block_on(Approach4Renderer::from_adapter(
                adapter.clone(),
                format,
                &self.meshes,
                &self.surfels,
            ))
            .expect("init renderer");
        let mut cfg = surface
            .get_default_config(&adapter, sz.width.max(1), sz.height.max(1))
            .expect("default surface config");
        cfg.format = format;
        cfg.present_mode = wgpu::PresentMode::AutoVsync;
        cfg.alpha_mode = caps.alpha_modes[0];
        cfg.desired_maximum_frame_latency = 2;
        surface.configure(&renderer.device, &cfg);
        println!(
            "[Unified] Window GPU: {} ({}) | Format: {:?} | Connecting to {}",
            renderer.adapter_name, renderer.adapter_backend, format, self.args.server
        );

        match self.rt.block_on(QuicClientHandle::connect(
            self.args.server,
            &self.args.name,
            self.args.team,
        )) {
            Ok(q) => {
                println!("[Unified] Connected live to {}!", self.args.server);
                self.quic = Some(q);
            }
            Err(e) => {
                eprintln!("[Unified] QUIC connect failed: {}", e);
            }
        }

        self.window = Some(win);
        self.surface = Some(surface);
        self.surface_cfg = Some(cfg);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                if let (Some(s), Some(cfg), Some(r)) =
                    (self.surface.as_ref(), self.surface_cfg.as_mut(), self.renderer.as_ref())
                {
                    cfg.width = sz.width.max(1);
                    cfg.height = sz.height.max(1);
                    s.configure(&r.device, cfg);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let nx = position.x as f32;
                let ny = position.y as f32;
                if self.mouse_mid || ((self.planet_mode || self.macro_planet_mode) && self.mouse_right) {
                    if let Some((lx, ly)) = self.last_mouse_drag {
                        if self.macro_planet_mode {
                            self.macro_planet_camera.drag_orbit(nx - lx, ny - ly);
                        } else if self.planet_mode {
                            self.planet_camera.drag_orbit(nx - lx, ny - ly);
                        } else {
                            self.camera.yaw -= (nx - lx) * 0.006;
                            self.camera.pitch = (self.camera.pitch + (ny - ly) * 0.005).clamp(0.22, 1.32);
                            self.camera_preset_easing = false;
                        }
                    }
                    self.last_mouse_drag = Some((nx, ny));
                } else {
                    self.last_mouse_drag = None;
                }
                self.mouse_pos = (nx, ny);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if down && !self.mouse_left {
                            self.input_dirty = true;
                            self.actuator_edge_fire = true;
                        }
                        self.mouse_left = down;
                    }
                    MouseButton::Right => {
                        if down && !self.mouse_right {
                            self.input_dirty = true;
                            self.actuator_edge_fire = true;
                        }
                        self.mouse_right = down;
                    }
                    MouseButton::Middle => {
                        self.mouse_mid = down;
                        if !down {
                            self.last_mouse_drag = None;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 4.5,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.08,
                };
                if self.macro_planet_mode {
                    self.macro_planet_camera.zoom(dy);
                } else if self.planet_mode {
                    self.planet_camera.zoom(
                        dy,
                        self.planet_world.r_core,
                        self.planet_world.depth as f32,
                    );
                } else {
                    let max_zoom = if self.z16_active && self.z16_world.depth >= 48 {
                        240.0
                    } else {
                        180.0
                    };
                    self.camera.dist = (self.camera.dist - dy).clamp(14.0, max_zoom);
                    self.camera_preset_easing = false;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if self.chat_open {
                        if event.state == ElementState::Pressed {
                            match code {
                                KeyCode::Escape => {
                                    self.chat_open = false;
                                    self.chat_input.clear();
                                }
                                KeyCode::Tab => {
                                    if !event.repeat {
                                        self.chat_shout = !self.chat_shout;
                                    }
                                }
                                KeyCode::Backspace => {
                                    self.chat_input.pop();
                                }
                                KeyCode::Enter | KeyCode::NumpadEnter => {
                                    let msg = self.chat_input.trim().to_string();
                                    if !msg.is_empty() {
                                        if let Some(ref q) = self.quic {
                                            let vol = if self.chat_shout { 1.0 } else { 0.25 };
                                            q.send_reliable_detached(ClientReliable::ChatCommand {
                                                text: msg,
                                                volume: vol,
                                            });
                                        }
                                    }
                                    self.chat_input.clear();
                                    self.chat_open = false;
                                }
                                _ => {
                                    if let Some(ref txt) = event.text {
                                        for ch in txt.chars() {
                                            if ch.is_ascii() && !ch.is_ascii_control() && self.chat_input.len() < 64 {
                                                self.chat_input.push(ch);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return;
                    }
                    if event.state == ElementState::Pressed {
                        self.keys.insert(code);
                        if !event.repeat {
                            if matches!(code, KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD | KeyCode::Space | KeyCode::KeyE | KeyCode::KeyR | KeyCode::KeyC) {
                                self.input_dirty = true;
                            }
                            if matches!(code, KeyCode::KeyE | KeyCode::KeyR) {
                                self.actuator_edge_fire = true;
                            }
                            match code {
                                KeyCode::Escape => event_loop.exit(),
                                KeyCode::Enter | KeyCode::NumpadEnter => {
                                    self.chat_open = true;
                                    self.chat_input.clear();
                                    self.keys.clear();
                                }
                                KeyCode::Digit1 => self.selected_element = 0,
                                KeyCode::Digit2 => self.selected_element = 1,
                                KeyCode::Digit3 => self.selected_element = 2,
                                KeyCode::BracketLeft => {
                                    self.focus_radius = (self.focus_radius - 0.5).clamp(2.0, 8.0);
                                }
                                KeyCode::BracketRight => {
                                    self.focus_radius = (self.focus_radius + 0.5).clamp(2.0, 8.0);
                                }
                                KeyCode::KeyP => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::TogglePhysicsNative);
                                    }
                                }
                                KeyCode::KeyK => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::CycleTimeScale);
                                    }
                                }
                                KeyCode::KeyJ => {
                                    if let Some(ref q) = self.quic {
                                        let shift_held = self.keys.contains(&KeyCode::ShiftLeft)
                                            || self.keys.contains(&KeyCode::ShiftRight);
                                        if shift_held {
                                            q.send_reliable_detached(ClientReliable::PlaceTimeDome {
                                                gx: 0.0,
                                                gy: 0.0,
                                                mode: 0,
                                            });
                                            self.next_dome_mode = 1;
                                        } else {
                                            let (gx, gy) = self.last_cursor_grid.unwrap_or_else(|| {
                                                self.pred_hero_pos
                                                    .map(|(hx, hy, _)| (hx, hy))
                                                    .unwrap_or((
                                                        self.world.grid_w as f32 * 0.5,
                                                        self.world.grid_h as f32 * 0.5,
                                                    ))
                                            });
                                            let mode = self.next_dome_mode;
                                            q.send_reliable_detached(ClientReliable::PlaceTimeDome {
                                                gx: gx.clamp(8.0, (self.world.grid_w.saturating_sub(8)) as f32),
                                                gy: gy.clamp(8.0, (self.world.grid_h.saturating_sub(8)) as f32),
                                                mode,
                                            });
                                            self.next_dome_mode = if mode == 1 { 2 } else { 1 };
                                            if let Some(ref audio) = self.audio {
                                                let sfx = if mode == 1 {
                                                    SfxKind::FrostCast
                                                } else {
                                                    SfxKind::FireCast
                                                };
                                                audio.trigger_sfx(sfx, mode, 0.55, 0.0);
                                            }
                                        }
                                    }
                                }
                                KeyCode::Space => self.pending_dash = true,
                                KeyCode::KeyC => self.pending_carve = true,
                                KeyCode::KeyE => self.pending_ability_e = true,
                                KeyCode::KeyR => self.pending_ult_r = true,
                                KeyCode::KeyT => self.style_clean = !self.style_clean,
                                KeyCode::KeyY => {
                                    self.z16_style_mode = (self.z16_style_mode + 1) % NUM_Z16_STYLES;
                                    println!(
                                        "[Unified] Switched client WGSL neural style -> {} [Y]",
                                        z16_style_label(self.z16_style_mode)
                                    );
                                }
                                KeyCode::KeyL => self.layer_slice_mode = (self.layer_slice_mode + 1) % 3,
                                KeyCode::KeyG => {
                                    self.show_squad_links = !self.show_squad_links;
                                    println!(
                                        "[Unified] Squad Cohesion Tether Arcs -> {} [G]",
                                        if self.show_squad_links { "ON" } else { "OFF" }
                                    );
                                }
                                KeyCode::KeyH => self.show_hud = !self.show_hud,
                                KeyCode::KeyU => {
                                    if let Some(ref audio) = self.audio {
                                        self.audio_muted = audio.toggle_mute();
                                    } else {
                                        self.audio_muted = !self.audio_muted;
                                    }
                                }
                                KeyCode::F4 => self.show_telemetry = !self.show_telemetry,
                                KeyCode::F1 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::SelectClass { class_id: CLASS_WARDEN });
                                    }
                                }
                                KeyCode::F2 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::SelectClass { class_id: CLASS_ALCHEMIST });
                                    }
                                }
                                KeyCode::F3 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::SelectClass { class_id: CLASS_RANGER });
                                    }
                                }
                                KeyCode::F8 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::SelectClass { class_id: 3 });
                                    }
                                }
                                KeyCode::F10 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::SelectClass {
                                            class_id: CLASS_BURROWER,
                                        });
                                    }
                                }
                                KeyCode::KeyM => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::ToggleSimMode);
                                    }
                                }
                                KeyCode::KeyN => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::ToggleNeuralBots);
                                    }
                                }
                                KeyCode::KeyB => {
                                    let count = self
                                        .quic
                                        .as_ref()
                                        .and_then(|q| q.state.lock().ok())
                                        .and_then(|st| st.entities.as_ref().map(|e| e.bot_brains.len()))
                                        .unwrap_or(0);
                                    if !self.brain_inspector_on {
                                        self.brain_inspector_on = true;
                                        self.brain_selected_idx = None;
                                    } else if count == 0 {
                                        self.brain_inspector_on = false;
                                        self.brain_selected_idx = None;
                                    } else {
                                        match self.brain_selected_idx {
                                            None => self.brain_selected_idx = Some(0),
                                            Some(idx) if idx + 1 < count => {
                                                self.brain_selected_idx = Some(idx + 1);
                                            }
                                            Some(_) => {
                                                self.brain_inspector_on = false;
                                                self.brain_selected_idx = None;
                                            }
                                        }
                                    }
                                }
                                KeyCode::KeyV | KeyCode::F6 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::CycleWeather);
                                    }
                                }
                                KeyCode::F5 => {
                                    let shift_held = self.keys.contains(&KeyCode::ShiftLeft)
                                        || self.keys.contains(&KeyCode::ShiftRight);
                                    if shift_held {
                                        if let Some(ref q) = self.quic {
                                            q.send_reliable_detached(ClientReliable::ResetWorld);
                                        }
                                    } else if self.macro_planet_mode {
                                        self.macro_planet_camera.cycle_mode();
                                        println!(
                                            "[Unified-MacroPlanet] Camera preset -> {} [F5]",
                                            self.macro_planet_camera.mode.label()
                                        );
                                    } else if self.planet_mode {
                                        self.planet_camera.cycle_mode(
                                            self.planet_world.r_core,
                                            self.planet_world.depth as f32,
                                        );
                                        println!(
                                            "[Unified-Planet] Camera preset -> {} [F5]",
                                            self.planet_camera.mode.label()
                                        );
                                    } else {
                                        self.camera_preset = self.camera_preset.next();
                                        self.camera_preset_easing = true;
                                        println!(
                                            "[Unified] Camera preset -> {} [F5]",
                                            self.camera_preset.label()
                                        );
                                    }
                                }
                                KeyCode::F7 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::CycleBiome);
                                        println!("[Unified] Cycling 3D server biome [F7]");
                                    }
                                }
                                KeyCode::F9 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::CycleEvolutionSpeed);
                                        println!("[Unified] Cycling Evolving World speed [F9]");
                                    }
                                }
                                KeyCode::F11 => {
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::CycleWorldLaw);
                                        println!("[Unified] Cycling World Law preset [F11]");
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else {
                        self.keys.remove(&code);
                        if matches!(code, KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD) {
                            self.input_dirty = true;
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if let Some(ref w) = self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.headless {
        run_headless(args)
    } else {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = InteractiveApp::new(args);
        event_loop.run_app(&mut app)?;
        Ok(())
    }
}
