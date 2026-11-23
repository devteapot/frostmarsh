//! Iteration 8B: Head-to-Head True 3D `Z=16` Direct-3D-Texture WGSL Renderer Sandbox
//! (`Sparse 8x8x8 Brick Raymarching` vs. `Smooth Surface-Nets + Volumetric Overlay`)
//! + Continuous `∇solid_density` 3D Capsule Collision Verification.
//!
//! Supports:
//! - `--headless`: Benchmarks Renderer A & B at 1080p and 4K on Metal, runs the 4-scenario
//!   continuous 3D capsule collision audit, writes `models/RENDER_3D_HEADTOHEAD_METRICS.json`,
//!   and composites the 1920×1080 2×2 visual showcase into `models/render_3d_headtohead_viz.png`.
//! - Interactive window (default): WASD fly/walk camera, mouse-drag orbit, `1`/`2` camera presets,
//!   and `G` live hot-toggle between Renderer A (Raymarch) and Renderer B (Surface-Nets)!

use clap::Parser;
use frostmarsh_3d_sandbox::camera::Camera3D;
use frostmarsh_3d_sandbox::collision::{run_capsule_audit, CapsuleAuditMetrics};
use frostmarsh_3d_sandbox::hero_mesh::{build_hero_knight_mesh, showcase_hero_instances};
use frostmarsh_3d_sandbox::renderer::{HeadToHeadRenderer, RenderMode};
use frostmarsh_3d_sandbox::surface_nets_3d::SurfaceNetsWorld;
use frostmarsh_3d_sandbox::volume::WorldVolume3D;
use glam::Vec3;
use image::{Rgba, RgbaImage};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

#[derive(Parser, Debug)]
#[command(name = "frostmarsh-3d-sandbox")]
#[command(about = "Head-to-Head True 3D Z=16 Direct-3D-Texture WGSL Renderer Benchmark & Sandbox")]
struct Args {
    /// Run headless benchmark + 2x2 showcase PNG + metrics JSON
    #[arg(long)]
    headless: bool,

    /// Optional external FM3D binary volume path (defaults to rich procedural 256x256x16 scene)
    #[arg(long)]
    volume: Option<PathBuf>,

    /// Output path for the 1920x1080 2x2 head-to-head showcase PNG
    #[arg(long, default_value = "../../models/render_3d_headtohead_viz.png")]
    out_png: PathBuf,

    /// Output path for the head-to-head benchmark metrics JSON
    #[arg(long, default_value = "../../models/RENDER_3D_HEADTOHEAD_METRICS.json")]
    out_json: PathBuf,

    /// Benchmark iterations per resolution & renderer
    #[arg(long, default_value_t = 30)]
    iters: usize,
}

#[derive(Serialize)]
struct ResBenchMetrics {
    p50_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    fps_p50: f64,
}

#[derive(Serialize)]
struct RendererMetrics {
    res_1080p: ResBenchMetrics,
    res_4k: ResBenchMetrics,
    cave_1080p: ResBenchMetrics,
}

#[derive(Serialize)]
struct SurfaceNetsBuildMetrics {
    full_build_ms: f64,
    dirty_brick_16x16x16_p50_ms: f64,
    dirty_brick_16x16x16_max_ms: f64,
    vertex_count: usize,
    triangle_count: usize,
    brick_count: usize,
}

#[derive(Serialize)]
struct HeadToHeadMetricsReport {
    adapter: String,
    backend: String,
    volume_dims: [usize; 3],
    axis_convention: &'static str,
    brick_occupancy_8x8x8: BrickStats,
    surface_nets_build: SurfaceNetsBuildMetrics,
    renderer_a_raymarch: RendererMetrics,
    renderer_b_surface_nets: RendererMetrics,
    speedup_b_over_a_1080p: f64,
    speedup_b_over_a_4k: f64,
    visual_parity_mae_rgb_255: VisualParityMetrics,
    capsule_collision_3d: CapsuleAuditMetrics,
}

#[derive(Serialize)]
struct BrickStats {
    grid_dims: [usize; 3],
    non_empty_bricks: usize,
    total_bricks: usize,
    occupancy_pct: f64,
}

#[derive(Serialize)]
struct VisualParityMetrics {
    bridge_view_mae: f64,
    cave_view_mae: f64,
}

// Crisp built-in 5x7 bitmap font for clean non-overlapping quadrant labels on the 2x2 showcase PNG!
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '/' => [0b00001, 0b00010, 0b00100, 0b00100, 0b01000, 0b10000, 0b00000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        _ => [0; 7],
    }
}

fn draw_badge(img: &mut RgbaImage, x0: u32, y0: u32, text: &str, accent: [u8; 3]) {
    let scale = 2u32;
    let char_w = 6 * scale;
    let pad_x = 12u32;
    let pad_y = 7u32;
    let box_w = text.len() as u32 * char_w + pad_x * 2;
    let box_h = 7 * scale + pad_y * 2;
    let (iw, ih) = img.dimensions();

    for py in y0..(y0 + box_h).min(ih) {
        for px in x0..(x0 + box_w).min(iw) {
            let p = img.get_pixel_mut(px, py);
            let is_bar = px < x0 + 4;
            if is_bar {
                *p = Rgba([accent[0], accent[1], accent[2], 255]);
            } else {
                // Dark translucent frosted HUD bar
                p[0] = ((p[0] as u32 * 55 + 14 * 200) / 255) as u8;
                p[1] = ((p[1] as u32 * 55 + 18 * 200) / 255) as u8;
                p[2] = ((p[2] as u32 * 55 + 28 * 200) / 255) as u8;
                p[3] = 255;
            }
        }
    }

    for (ci, ch) in text.chars().enumerate() {
        let rows = glyph_5x7(ch);
        let cx = x0 + pad_x + ci as u32 * char_w;
        let cy = y0 + pad_y;
        for (ry, &bits) in rows.iter().enumerate() {
            for rx in 0..5u32 {
                if (bits >> (4 - rx)) & 1 == 1 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + rx * scale + sx;
                            let py = cy + ry as u32 * scale + sy;
                            if px < iw && py < ih {
                                img.put_pixel(px, py, Rgba([244, 248, 255, 255]));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn blit_quadrant(canvas: &mut RgbaImage, rgba: &[u8], qw: u32, qh: u32, ox: u32, oy: u32) {
    for y in 0..qh {
        let row = (y * qw * 4) as usize;
        for x in 0..qw {
            let i = row + (x * 4) as usize;
            canvas.put_pixel(ox + x, oy + y, Rgba([rgba[i], rgba[i + 1], rgba[i + 2], 255]));
        }
    }
}

fn mae_rgb(a: &[u8], b: &[u8]) -> f64 {
    let pixels = (a.len() / 4).min(b.len() / 4);
    let mut sum = 0u64;
    for i in 0..pixels {
        let o = i * 4;
        sum += (a[o] as i32 - b[o] as i32).unsigned_abs() as u64;
        sum += (a[o + 1] as i32 - b[o + 1] as i32).unsigned_abs() as u64;
        sum += (a[o + 2] as i32 - b[o + 2] as i32).unsigned_abs() as u64;
    }
    sum as f64 / (pixels * 3).max(1) as f64
}

fn load_or_generate_volume(path: Option<&Path>) -> Result<WorldVolume3D, Box<dyn std::error::Error>> {
    if let Some(p) = path {
        println!("[3D-Sandbox] Loading FM3D volume from {}", p.display());
        WorldVolume3D::load_fm3d(p)
    } else {
        println!("[3D-Sandbox] Generating procedural isotropic 256x256x16 alpine/bridge/cave/waterfall/wildfire volume...");
        let t0 = Instant::now();
        let vol = WorldVolume3D::generate_showcase();
        println!(
            "[3D-Sandbox] Procedural 256x256x16 volume generated in {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(vol)
    }
}

fn run_headless(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let volume = load_or_generate_volume(args.volume.as_deref())?;
    let bricks = volume.build_brick_occupancy();

    // 1. Build Smooth Surface-Nets & benchmark single 16x16x16 dirty-brick rebuilds!
    let (mut sn_world, sn_full_ms) = SurfaceNetsWorld::build_all(&volume);
    let mut dirty_samples = Vec::with_capacity(20);
    for &(bx, bz) in &[
        (7, 7),
        (8, 7),
        (8, 8),
        (9, 8),
        (6, 9),
        (5, 9),
        (8, 6),
        (9, 7),
        (7, 8),
        (5, 10),
    ] {
        for _ in 0..2 {
            dirty_samples.push(sn_world.rebuild_brick(&volume, bx, bz));
        }
    }
    dirty_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dirty_p50 = dirty_samples[dirty_samples.len() / 2];
    let dirty_max = *dirty_samples.last().unwrap_or(&0.0);
    println!(
        "[Surface-Nets] Full 256x16x256 build: {:.2} ms | 16x16x16 dirty brick p50: {:.3} ms (max {:.3} ms) | {} verts, {} tris",
        sn_full_ms,
        dirty_p50,
        dirty_max,
        sn_world.total_vertices(),
        sn_world.total_triangles()
    );

    // 2. Run continuous ∇solid_density 3D capsule collision audit
    let capsule_audit = run_capsule_audit(&volume);
    println!(
        "[Capsule-3D] Bridge={} (y={:.2}..{:.2}, pen={:.4}m) | UnderBridge={} (clr={:.2}m, pen={:.4}m) | Cave={} (clr={:.2}m, pen={:.4}m) | 45degWallSlide={} (v_tan={:.2}m/s, pen={:.4}m)",
        capsule_audit.bridge_walk_ok,
        capsule_audit.bridge_min_y,
        capsule_audit.bridge_max_y,
        capsule_audit.bridge_max_penetration_m,
        capsule_audit.under_bridge_walk_ok,
        capsule_audit.under_bridge_min_clearance_m,
        capsule_audit.under_bridge_max_penetration_m,
        capsule_audit.cave_traverse_ok,
        capsule_audit.cave_min_ceiling_clearance_m,
        capsule_audit.cave_max_penetration_m,
        capsule_audit.wall_slide_45deg_ok,
        capsule_audit.wall_slide_min_tangent_speed,
        capsule_audit.wall_slide_max_penetration_m
    );

    // 3. Init GPU Head-to-Head Renderer
    let hero_mesh = build_hero_knight_mesh();
    let hero_instances = showcase_hero_instances();
    let mut renderer = rt.block_on(HeadToHeadRenderer::new_headless(
        &volume,
        &bricks,
        &sn_world,
        &hero_mesh,
        &hero_instances,
    ))?;
    println!(
        "[GPU] Adapter: {} ({})",
        renderer.adapter_name, renderer.adapter_backend
    );

    let cam_bridge = Camera3D::preset_bridge_waterfall();
    let cam_cave = Camera3D::preset_cave_mouth();

    // 4. Benchmark BOTH renderers at 1920x1080 and 3840x2160 (4K)
    let warmup = 6;
    let iters = args.iters.max(10);

    let a_1080 = renderer.benchmark(1920, 1080, RenderMode::Raymarch, &cam_bridge, warmup, iters);
    let a_4k = renderer.benchmark(3840, 2160, RenderMode::Raymarch, &cam_bridge, warmup, iters);
    let a_cave = renderer.benchmark(1920, 1080, RenderMode::Raymarch, &cam_cave, warmup, iters);

    let b_1080 = renderer.benchmark(1920, 1080, RenderMode::SurfaceNets, &cam_bridge, warmup, iters);
    let b_4k = renderer.benchmark(3840, 2160, RenderMode::SurfaceNets, &cam_bridge, warmup, iters);
    let b_cave = renderer.benchmark(1920, 1080, RenderMode::SurfaceNets, &cam_cave, warmup, iters);

    println!(
        "[Bench 1080p Bridge] A Raymarch: p50={:.2}ms p99={:.2}ms ({:.1} FPS) | B Surface-Nets: p50={:.2}ms p99={:.2}ms ({:.1} FPS)",
        a_1080.p50_ms, a_1080.p99_ms, a_1080.fps_p50,
        b_1080.p50_ms, b_1080.p99_ms, b_1080.fps_p50
    );
    println!(
        "[Bench 4K    Bridge] A Raymarch: p50={:.2}ms p99={:.2}ms ({:.1} FPS) | B Surface-Nets: p50={:.2}ms p99={:.2}ms ({:.1} FPS)",
        a_4k.p50_ms, a_4k.p99_ms, a_4k.fps_p50,
        b_4k.p50_ms, b_4k.p99_ms, b_4k.fps_p50
    );
    println!(
        "[Bench 1080p Cave  ] A Raymarch: p50={:.2}ms p99={:.2}ms ({:.1} FPS) | B Surface-Nets: p50={:.2}ms p99={:.2}ms ({:.1} FPS)",
        a_cave.p50_ms, a_cave.p99_ms, a_cave.fps_p50,
        b_cave.p50_ms, b_cave.p99_ms, b_cave.fps_p50
    );

    // 5. Render 1920x1080 2x2 Head-to-Head Visual Showcase (each quadrant 960x540)
    let qw = 960u32;
    let qh = 540u32;
    let tl = renderer.render_to_rgba(qw, qh, RenderMode::Raymarch, &cam_bridge, 1.5);
    let tr = renderer.render_to_rgba(qw, qh, RenderMode::SurfaceNets, &cam_bridge, 1.5);
    let bl = renderer.render_to_rgba(qw, qh, RenderMode::Raymarch, &cam_cave, 1.5);
    let br = renderer.render_to_rgba(qw, qh, RenderMode::SurfaceNets, &cam_cave, 1.5);

    let bridge_mae = mae_rgb(&tl, &tr);
    let cave_mae = mae_rgb(&bl, &br);

    let mut canvas = RgbaImage::new(1920, 1080);
    blit_quadrant(&mut canvas, &tl, qw, qh, 0, 0);
    blit_quadrant(&mut canvas, &tr, qw, qh, qw, 0);
    blit_quadrant(&mut canvas, &bl, qw, qh, 0, qh);
    blit_quadrant(&mut canvas, &br, qw, qh, qw, qh);

    // Crisp 2px dark divider crosshair between quadrants
    for x in 0..1920 {
        canvas.put_pixel(x, qh - 1, Rgba([18, 22, 32, 255]));
        canvas.put_pixel(x, qh, Rgba([18, 22, 32, 255]));
    }
    for y in 0..1080 {
        canvas.put_pixel(qw - 1, y, Rgba([18, 22, 32, 255]));
        canvas.put_pixel(qw, y, Rgba([18, 22, 32, 255]));
    }

    draw_badge(
        &mut canvas,
        14,
        12,
        &format!(
            "A: RAYMARCH (8X8X8 BRICK DDA) | 1080P {:.2}MS ({:.0} FPS) | 4K {:.2}MS",
            a_1080.p50_ms, a_1080.fps_p50, a_4k.p50_ms
        ),
        [68, 178, 255],
    );
    draw_badge(
        &mut canvas,
        qw + 14,
        12,
        &format!(
            "B: SURFACE-NETS + VOL OVERLAY | 1080P {:.2}MS ({:.0} FPS) | 4K {:.2}MS",
            b_1080.p50_ms, b_1080.fps_p50, b_4k.p50_ms
        ),
        [92, 232, 152],
    );
    draw_badge(
        &mut canvas,
        14,
        qh + 12,
        &format!(
            "A: RAYMARCH - VAULTED BASALT CAVE + MAGMA | 1080P {:.2}MS ({:.0} FPS)",
            a_cave.p50_ms, a_cave.fps_p50
        ),
        [255, 168, 68],
    );
    draw_badge(
        &mut canvas,
        qw + 14,
        qh + 12,
        &format!(
            "B: SURFACE-NETS - VAULTED CAVE ({:.2}MS DIRTY BRICK) | 1080P {:.2}MS",
            dirty_p50, b_cave.p50_ms
        ),
        [92, 232, 152],
    );

    if let Some(parent) = args.out_png.parent() {
        std::fs::create_dir_all(parent)?;
    }
    canvas.save(&args.out_png)?;
    println!("[Output] Saved 1920x1080 2x2 showcase PNG to {}", args.out_png.display());

    // 6. Save structured metrics JSON
    let report = HeadToHeadMetricsReport {
        adapter: renderer.adapter_name.clone(),
        backend: renderer.adapter_backend.clone(),
        volume_dims: [volume.width, volume.height, volume.depth],
        axis_convention: "World right-handed Y-up: X=width(256), Y=depth/altitude(16), Z=height(256); GPU TextureDimension::D3 Extent3d(256, 16, 256)",
        brick_occupancy_8x8x8: BrickStats {
            grid_dims: [bricks.bx_count, bricks.by_count, bricks.bz_count],
            non_empty_bricks: bricks.non_empty_bricks,
            total_bricks: bricks.total_bricks,
            occupancy_pct: 100.0 * bricks.non_empty_bricks as f64 / bricks.total_bricks.max(1) as f64,
        },
        surface_nets_build: SurfaceNetsBuildMetrics {
            full_build_ms: sn_full_ms,
            dirty_brick_16x16x16_p50_ms: dirty_p50,
            dirty_brick_16x16x16_max_ms: dirty_max,
            vertex_count: sn_world.total_vertices(),
            triangle_count: sn_world.total_triangles(),
            brick_count: sn_world.bricks.len(),
        },
        renderer_a_raymarch: RendererMetrics {
            res_1080p: ResBenchMetrics {
                p50_ms: a_1080.p50_ms,
                p99_ms: a_1080.p99_ms,
                mean_ms: a_1080.mean_ms,
                fps_p50: a_1080.fps_p50,
            },
            res_4k: ResBenchMetrics {
                p50_ms: a_4k.p50_ms,
                p99_ms: a_4k.p99_ms,
                mean_ms: a_4k.mean_ms,
                fps_p50: a_4k.fps_p50,
            },
            cave_1080p: ResBenchMetrics {
                p50_ms: a_cave.p50_ms,
                p99_ms: a_cave.p99_ms,
                mean_ms: a_cave.mean_ms,
                fps_p50: a_cave.fps_p50,
            },
        },
        renderer_b_surface_nets: RendererMetrics {
            res_1080p: ResBenchMetrics {
                p50_ms: b_1080.p50_ms,
                p99_ms: b_1080.p99_ms,
                mean_ms: b_1080.mean_ms,
                fps_p50: b_1080.fps_p50,
            },
            res_4k: ResBenchMetrics {
                p50_ms: b_4k.p50_ms,
                p99_ms: b_4k.p99_ms,
                mean_ms: b_4k.mean_ms,
                fps_p50: b_4k.fps_p50,
            },
            cave_1080p: ResBenchMetrics {
                p50_ms: b_cave.p50_ms,
                p99_ms: b_cave.p99_ms,
                mean_ms: b_cave.mean_ms,
                fps_p50: b_cave.fps_p50,
            },
        },
        speedup_b_over_a_1080p: a_1080.p50_ms / b_1080.p50_ms.max(1e-4),
        speedup_b_over_a_4k: a_4k.p50_ms / b_4k.p50_ms.max(1e-4),
        visual_parity_mae_rgb_255: VisualParityMetrics {
            bridge_view_mae: bridge_mae,
            cave_view_mae: cave_mae,
        },
        capsule_collision_3d: capsule_audit,
    };

    if let Some(parent) = args.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_json, serde_json::to_string_pretty(&report)?)?;
    println!("[Output] Saved metrics JSON to {}", args.out_json.display());
    Ok(())
}

// =========================================================================
// Interactive Window Mode (WASD Fly/Walk, Mouse Orbit, G-Key Live Toggle!)
// =========================================================================

struct InteractiveApp {
    args: Args,
    rt: tokio::runtime::Runtime,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_cfg: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<HeadToHeadRenderer>,
    mode: RenderMode,
    cam: Camera3D,
    keys: HashSet<KeyCode>,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    start_time: Instant,
    last_frame: Instant,
}

impl InteractiveApp {
    fn new(args: Args) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("tokio rt");
        Self {
            args,
            rt,
            window: None,
            surface: None,
            surface_cfg: None,
            renderer: None,
            mode: RenderMode::Raymarch,
            cam: Camera3D::preset_bridge_waterfall(),
            keys: HashSet::new(),
            dragging: false,
            last_cursor: None,
            start_time: Instant::now(),
            last_frame: Instant::now(),
        }
    }

    fn update_title(&self) {
        if let Some(ref w) = self.window {
            w.set_title(&format!(
                "Frostmarsh 3D Sandbox [G: Toggle] — {} | WASD Move, Drag Orbit, 1/2 Presets",
                self.mode.label()
            ));
        }
    }

    fn render_frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().clamp(0.001, 0.1);
        self.last_frame = now;

        // WASD + Q/E free-fly camera movement
        let speed = if self.keys.contains(&KeyCode::ShiftLeft) { 22.0 } else { 10.0 };
        let fwd = self.cam.forward();
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        if self.keys.contains(&KeyCode::KeyW) {
            self.cam.pos += fwd * speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam.pos -= fwd * speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam.pos += right * speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam.pos -= right * speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyE) || self.keys.contains(&KeyCode::Space) {
            self.cam.pos.y += speed * dt;
        }
        if self.keys.contains(&KeyCode::KeyQ) {
            self.cam.pos.y -= speed * dt;
        }

        let (Some(surface), Some(cfg), Some(renderer)) = (
            self.surface.as_ref(),
            self.surface_cfg.as_ref(),
            self.renderer.as_mut(),
        ) else {
            return;
        };

        let surf_tex = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = surf_tex.texture.create_view(&Default::default());
        let sim_t = self.start_time.elapsed().as_secs_f32();
        renderer.render_into(&view, cfg.width, cfg.height, self.mode, &self.cam, sim_t);
        renderer.queue.present(surf_tex);
    }
}

impl ApplicationHandler for InteractiveApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Frostmarsh 3D Sandbox — Loading...")
            .with_inner_size(LogicalSize::new(1440.0, 900.0));
        let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let sz = win.inner_size();

        let volume = load_or_generate_volume(self.args.volume.as_deref()).expect("load volume");
        let bricks = volume.build_brick_occupancy();
        let (sn_world, _) = SurfaceNetsWorld::build_all(&volume);
        let hero_mesh = build_hero_knight_mesh();
        let hero_instances = showcase_hero_instances();

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
            .expect("adapter");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let renderer = self
            .rt
            .block_on(HeadToHeadRenderer::from_adapter(
                adapter.clone(),
                format,
                &volume,
                &bricks,
                &sn_world,
                &hero_mesh,
                &hero_instances,
            ))
            .expect("init renderer");

        let mut cfg = surface
            .get_default_config(&adapter, sz.width.max(1), sz.height.max(1))
            .expect("surface config");
        cfg.format = format;
        cfg.present_mode = wgpu::PresentMode::AutoVsync;
        cfg.alpha_mode = caps.alpha_modes[0];
        surface.configure(&renderer.device, &cfg);

        self.window = Some(win);
        self.surface = Some(surface);
        self.surface_cfg = Some(cfg);
        self.renderer = Some(renderer);
        self.update_title();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                if let (Some(surface), Some(cfg), Some(renderer)) = (
                    self.surface.as_ref(),
                    self.surface_cfg.as_mut(),
                    self.renderer.as_ref(),
                ) {
                    cfg.width = sz.width.max(1);
                    cfg.height = sz.height.max(1);
                    surface.configure(&renderer.device, cfg);
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = self.last_cursor {
                    if self.dragging {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.cam.yaw += dx * 0.005;
                        self.cam.pitch = (self.cam.pitch - dy * 0.005).clamp(-1.45, 1.45);
                    }
                }
                self.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys.insert(code);
                        if !event.repeat {
                            match code {
                                KeyCode::Escape => event_loop.exit(),
                                KeyCode::KeyG => {
                                    self.mode = self.mode.toggle();
                                    self.update_title();
                                    println!("[3D-Sandbox] Switched to {}", self.mode.label());
                                }
                                KeyCode::Digit1 => self.cam = Camera3D::preset_bridge_waterfall(),
                                KeyCode::Digit2 => self.cam = Camera3D::preset_cave_mouth(),
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
