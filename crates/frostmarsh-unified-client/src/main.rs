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
mod mesh_gen;
mod net;
mod palette;
mod parametric;
mod renderer;
mod surfel_gen;
mod world_bridge;

use crate::audio::{SfxKind, SpatialAudioEngine};
use crate::camera::MobaCamera;
use crate::hud::{build_hud, HudConfig};
use crate::mesh_gen::StaticMeshBank;
use crate::net::QuicClientHandle;
use crate::renderer::{Approach4Renderer, Atmosphere, RenderStyle};
use crate::surfel_gen::SurfelProtoBank;
use crate::world_bridge::{assemble_live_frame, LiveApproach4World};
use clap::Parser;
use frostmarsh_proto::*;
use glam::Vec3;
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
    #[arg(long, default_value = ".")]
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
    println!("[Unified] Synced {}/{} chunks (entities={})", rc, tot, has_ent);

    let (gw, gh, gd) = {
        let st = quic.state.lock().unwrap();
        (st.width, st.height, st.depth)
    };
    let mut world = LiveApproach4World::new(gw, gh, gd);

    // Full initial 3D volumetric build (L1 surface + L2 highland/skybridges/waterfalls + L3 plumes)
    let dirty_init = {
        let mut st = quic.state.lock().unwrap();
        st.take_dirty_list()
    };
    let aplus_copy = { quic.state.lock().unwrap().aplus_layers.clone() };
    let (_, full_sync_build_ms) = world.update_dirty_chunks(&aplus_copy, &dirty_init);

    // Observe 10 Hz incremental 3D dirty-chunk rebuilds
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

    renderer.upload_minimap(world.grid_w as u32, world.grid_h as u32, &world.minimap_rgba);

    // Heroic 3D Volumetric Vista showcasing Layer 2 Highland Plateaus, Plunging 3D Waterfalls,
    // Overhanging Stone Sky-Arch (with fjord underneath!), Central Spire, and Layer 3 Plumes!
    let focus_target = Vec3::new(0.0, 2.2, -(gh as f32) * 0.14);

    let cam_main = MobaCamera {
        target: focus_target,
        yaw: 0.48,
        pitch: 0.56,
        dist: 102.0,
        fov_y: 38.0_f32.to_radians(),
        ortho: false,
    };
    let sun_dir = Approach4Renderer::sun_dir(atmo);
    let mats_main = cam_main.compute_matrices(1920.0, 1080.0, sun_dir);
    let sim_time = 2.4f32;
    let frame_main = assemble_live_frame(
        &world,
        entities.as_ref(),
        &surfels,
        &mats_main,
        cam_main.target,
        cam_main.dist,
        sim_time,
        0,
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
                fps: fps_1080,
                rtt_ms: 1.2,
                chat_open: false,
                chat_input: "",
                chat_shout: true,
                show_scoreboard: false,
                audio_muted: true,
                audio_available: false,
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

    // Capture 1: Crisp 1920×1080 Live Battle + Full HUD + Live 256×256 Minimap
    let style_main = style_from_name(&args.style, 1080);
    let hud_main = build_hud(
        1920.0,
        1080.0,
        &world,
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
            fps: fps_1080,
            rtt_ms: 1.2,
            chat_open: false,
            chat_input: "",
            chat_shout: true,
            show_scoreboard: false,
            audio_muted: true,
            audio_available: false,
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

    // Capture 2: 2×2 Multi-Angle Orbit Showcase (960×540 per panel -> 1920×1080)
    let views = [
        // Top-left: Wide 3D Shattered Reach overview (ALL 4 LAYERS with HUD)
        (
            MobaCamera { target: Vec3::new(0.0, 2.0, -(gh as f32) * 0.14), yaw: 0.44, pitch: 0.62, dist: 124.0, fov_y: 40.0_f32.to_radians(), ortho: false },
            RenderStyle::pixel_art(1),
            true,
            0u8,
        ),
        // Top-right: Low dramatic angle looking UNDER the Layer 2 Stone Sky-Arch + Plunging 3D Waterfalls!
        (
            MobaCamera { target: Vec3::new(-4.0, 2.5, -(gh as f32) * 0.23), yaw: 0.35, pitch: 0.36, dist: 56.0, fov_y: 36.0_f32.to_radians(), ortho: false },
            RenderStyle::pixel_art(1),
            false,
            0u8,
        ),
        // Bottom-left: Layer 2 + Layer 3 Slice ONLY (`L` slice mode = 2: floating Skybridges, Highland Lakes, Plumes!)
        (
            MobaCamera { target: Vec3::new(0.0, 4.0, -(gh as f32) * 0.12), yaw: 0.62, pitch: 0.56, dist: 112.0, fov_y: 38.0_f32.to_radians(), ortho: false },
            RenderStyle::pixel_art(1),
            false,
            2u8,
        ),
        // Bottom-right: Southern Timber Skybridge + SE Obsidian High-Overlook (Clean 3D Resolve)
        (
            MobaCamera { target: Vec3::new(14.0, 2.5, (gh as f32) * 0.18), yaw: -0.95, pitch: 0.46, dist: 68.0, fov_y: 38.0_f32.to_radians(), ortho: false },
            RenderStyle::clean_3d(),
            false,
            0u8,
        ),
    ];
    let mut panels = Vec::with_capacity(4);
    for (cam, st, with_hud, slice_mode) in &views {
        let m = cam.compute_matrices(960.0, 540.0, sun_dir);
        let fr = assemble_live_frame(&world, entities.as_ref(), &surfels, &m, cam.target, cam.dist, sim_time, *slice_mode);
        let hb = if *with_hud {
            Some(build_hud(
                960.0,
                540.0,
                &world,
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
                    fps: fps_1080,
                    rtt_ms: 1.2,
                    chat_open: false,
                    chat_input: "",
                    chat_shout: true,
                    show_scoreboard: false,
                    audio_muted: true,
                    audio_available: false,
                },
            ))
        } else {
            None
        };
        let (px, _, _) = renderer.render_to_rgba(960, 540, st, atmo, cam, &m, sim_time, &fr, hb.as_ref());
        panels.push(px);
    }
    let grid = compose_2x2_grid([&panels[0], &panels[1], &panels[2], &panels[3]], 960, 540);
    save_rgba_png(&args.out_dir.join("unified_orbit_4view.png"), 1920, 1080, &grid)?;

    let tel = entities.as_ref().map(|e| e.telemetry.clone());
    let mesh_inst_cnt: usize = frame_main.mesh_batches.iter().map(|b| b.len()).sum();
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
        terrain_triangles: frame_main.terrain_idxs.len() / 3,
        mesh_instances: mesh_inst_cnt,
        surfel_draws: frame_main.surfel_refs.len(),
        volumetric_splats: frame_main.vol_splats.len(),
        hud_vertices: hud_main.vertices.len(),
        gpu_benchmarks,
    };
    let json = serde_json::to_string_pretty(&metrics)?;
    std::fs::write(args.out_dir.join("unified_metrics.json"), &json)?;
    println!("[Unified] Saved unified_live_1080p_hud.png, unified_orbit_4view.png, unified_metrics.json");
    Ok(())
}

// =========================================================================
// INTERACTIVE PLAYABLE WINDOW MODE (macOS Metal / Linux Vulkan via winit)
// =========================================================================

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
    camera: MobaCamera,
    style_clean: bool,
    layer_slice_mode: u8,
    show_hud: bool,
    show_telemetry: bool,
    brain_inspector_on: bool,
    brain_selected_idx: Option<usize>,
    selected_element: u8,
    focus_radius: f32,
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
    last_input_send: Instant,
    last_actuator_send: Instant,
    start_time: Instant,
    last_frame: Instant,
    fps_ema: f32,
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
            camera: MobaCamera {
                target: Vec3::new(0.0, 1.0, 0.0),
                yaw: 0.42,
                pitch: 0.74,
                dist: 68.0,
                fov_y: 38.0_f32.to_radians(),
                ortho: false,
            },
            style_clean,
            layer_slice_mode: 0,
            show_hud: true,
            show_telemetry: true,
            brain_inspector_on: true,
            brain_selected_idx: None,
            selected_element: 0,
            focus_radius: 4.0,
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
            last_input_send: Instant::now(),
            last_actuator_send: Instant::now(),
            start_time: Instant::now(),
            last_frame: Instant::now(),
            fps_ema: 60.0,
        }
    }

    fn render_frame(&mut self) {
        let dt = self.last_frame.elapsed().as_secs_f32().clamp(0.0005, 0.1);
        self.last_frame = Instant::now();
        let inst_fps = 1.0 / dt;
        self.fps_ema = self.fps_ema * 0.90 + inst_fps * 0.10;
        let sim_time = self.start_time.elapsed().as_secs_f32();

        // Camera keyboard orbit (locked while typing in acoustic chat!)
        if !self.chat_open {
            if self.keys.contains(&KeyCode::KeyQ) {
                self.camera.yaw -= 1.6 * dt;
            }
            if self.keys.contains(&KeyCode::KeyZ) {
                self.camera.yaw += 1.6 * dt;
            }
        }

        // Pull latest 3D volumetric A+ dirty chunks + entities from QUIC
        let (entities, rtt_ms) = if let Some(ref quic) = self.quic {
            let (gw, gh, gd, dirty, layers_opt, ent, rtt) = {
                let mut st = quic.state.lock().unwrap();
                let d = st.take_dirty_list();
                let c = if !d.is_empty() { Some(st.aplus_layers.clone()) } else { None };
                (st.width, st.height, st.depth, d, c, st.entities.clone(), st.rtt_ms)
            };
            self.world.resize_if_needed(gw, gh, gd);
            if let Some(layers) = layers_opt {
                self.world.update_dirty_chunks(&layers, &dirty);
            }
            (ent, rtt)
        } else {
            (None, 0.0)
        };

        // Smooth 3D camera follow onto your hero's exact altitude
        if let Some(ref ent) = entities {
            if let Some(me) = ent.players.iter().find(|p| p.id == ent.your_player_id) {
                let (wx, wz) = self.world.grid_to_world_xz(me.x, me.y);
                let gy = self.world.sample_3d_y(me.x, me.y, me.z);
                let desired = Vec3::new(wx, gy + 0.8, wz);
                self.camera.target = self.camera.target.lerp(desired, (dt * 9.0).min(1.0));
            }
        }

        let (Some(surface), Some(cfg), Some(renderer)) =
            (self.surface.as_ref(), self.surface_cfg.as_ref(), self.renderer.as_mut())
        else {
            return;
        };

        if self.world.minimap_dirty {
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

        // Update Real-Time 3D Spatial Audio Engine (ambient waterfalls/wildfire/wind + spatial one-shots + SNR voice chirps)
        if let Some(ref audio) = self.audio {
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

            let (nearest_wf, nearby_fire) = self.world.sample_ambient_audio_sources(listener_pos);
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

        // Send 30 Hz inputs to server (camera-basis WASD + exact mouse-ray aiming!)
        if self.last_input_send.elapsed() >= Duration::from_millis(33) {
            self.last_input_send = Instant::now();
            if let Some(ref quic) = self.quic {
                let mut f_in = 0.0f32;
                let mut r_in = 0.0f32;
                if !self.chat_open {
                    if self.keys.contains(&KeyCode::KeyW) { f_in += 1.0; }
                    if self.keys.contains(&KeyCode::KeyS) { f_in -= 1.0; }
                    if self.keys.contains(&KeyCode::KeyD) { r_in += 1.0; }
                    if self.keys.contains(&KeyCode::KeyA) { r_in -= 1.0; }
                }

                // Project camera forward/right onto XZ grid plane
                let fwd_xz = Vec3::new(mats.fwd.x, 0.0, mats.fwd.z).normalize_or(Vec3::NEG_Z);
                let right_xz = Vec3::new(-fwd_xz.z, 0.0, fwd_xz.x);
                let move_v = fwd_xz * f_in + right_xz * r_in;

                // Aim toward mouse ground ray hit
                let mut aim_x = fwd_xz.x;
                let mut aim_y = fwd_xz.z;
                if let Some((hit_wx, hit_wz)) = self.camera.screen_to_world_xz(
                    &mats,
                    self.mouse_pos.0,
                    self.mouse_pos.1,
                    out_w as f32,
                    out_h as f32,
                    self.camera.target.y,
                ) {
                    let (hgx, hgy) = self.world.world_xz_to_grid(hit_wx, hit_wz);
                    if let Some(me) = entities
                        .as_ref()
                        .and_then(|e| e.players.iter().find(|p| p.id == e.your_player_id))
                    {
                        aim_x = hgx - me.x;
                        aim_y = hgy - me.y;
                    }
                }

                let physics_native = entities
                    .as_ref()
                    .map(|e| e.telemetry.physics_native)
                    .unwrap_or(false);

                self.input_seq = self.input_seq.wrapping_add(1);
                let now_ms = quic.state.lock().unwrap().client_time_ms();
                let shift_held = !self.chat_open
                    && (self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight));

                if physics_native && !self.chat_open {
                    let emit_matter = self.pending_ability_e
                        || self.keys.contains(&KeyCode::KeyE)
                        || (shift_held && self.mouse_left);
                    let emit_heat = self.mouse_left && !shift_held;
                    let emit_cryo = self.mouse_right;
                    let emit_pulse = self.pending_ult_r;

                    let ready_cont = self.last_actuator_send.elapsed() >= Duration::from_millis(85);
                    if emit_pulse || (ready_cont && (emit_heat || emit_cryo || emit_matter)) {
                        self.last_actuator_send = Instant::now();
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
                    }
                }

                let inp = ClientInputDatagram {
                    seq: self.input_seq,
                    move_x: move_v.x,
                    move_y: move_v.z,
                    aim_x,
                    aim_y,
                    dash: !self.chat_open && (self.pending_dash || self.keys.contains(&KeyCode::Space)),
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

        let style = if self.style_clean {
            RenderStyle::clean_3d()
        } else {
            RenderStyle::pixel_art(if out_h >= 1600 { 3 } else { 2 })
        };
        let frame = assemble_live_frame(
            &self.world,
            entities.as_ref(),
            &self.surfels,
            &mats,
            self.camera.target,
            self.camera.dist,
            sim_time,
            self.layer_slice_mode,
        );
        let hud_batch = build_hud(
            out_w as f32,
            out_h as f32,
            &self.world,
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
                fps: self.fps_ema,
                rtt_ms,
                chat_open: self.chat_open,
                chat_input: &self.chat_input,
                chat_shout: self.chat_shout,
                show_scoreboard: !self.chat_open && self.keys.contains(&KeyCode::Tab),
                audio_muted: self.audio_muted,
                audio_available: self.audio.is_some(),
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
                if self.mouse_mid {
                    if let Some((lx, ly)) = self.last_mouse_drag {
                        self.camera.yaw -= (nx - lx) * 0.006;
                        self.camera.pitch = (self.camera.pitch + (ny - ly) * 0.005).clamp(0.32, 1.32);
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
                    MouseButton::Left => self.mouse_left = down,
                    MouseButton::Right => self.mouse_right = down,
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
                self.camera.dist = (self.camera.dist - dy).clamp(26.0, 170.0);
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
                                KeyCode::Space => self.pending_dash = true,
                                KeyCode::KeyC => self.pending_carve = true,
                                KeyCode::KeyE => self.pending_ability_e = true,
                                KeyCode::KeyR => self.pending_ult_r = true,
                                KeyCode::KeyT => self.style_clean = !self.style_clean,
                                KeyCode::KeyL => self.layer_slice_mode = (self.layer_slice_mode + 1) % 3,
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
                                    if let Some(ref q) = self.quic {
                                        q.send_reliable_detached(ClientReliable::ResetWorld);
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else {
                        self.keys.remove(&code);
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
