//! Unified Scaled `wgpu 30.0.2` Renderer (Approach 4 + Tier 3 Hybrid Soft Vol-3DGS + GPU 2D HUD)
//!
//! Works BOTH interactively (`winit` Metal / Vulkan window) AND headlessly (`--headless` 1080p/4K).
//! 6-Stage GPU Pipeline:
//!   1. 2048×2048 Directional Shadow Map (`shadow_pass.wgsl`)
//!   2. 2× Supersampled Hard-Z Geometry (`mesh_pass.wgsl` + `surfel_instanced.wgsl` + 1 KB `flash()` palette)
//!   3. Depth-Tested Soft Volumetric 3DGS (`volumetric_splat.wgsl`, shared Hard-Z depth, 0 white clipping)
//!   4. Stage A Resolve (`pixel_resolve.wgsl`: 9-step Cool-Indigo/Warm-Cream Pixel-Art vs Clean 3D)
//!   5. Stage B 1-Tap Blit (`blit_pass.wgsl` -> native output)
//!   6. Native-Res GPU 2D Action-MMO HUD + Live 256×256 Tactical Minimap (`hud_pass.wgsl`)

use crate::camera::{CameraMatrices, MobaCamera};
use crate::hud::{HudBatch, HudVertex};
use crate::mesh_gen::{MeshInstance, MeshSlice, MeshVertex, StaticMeshBank, MESH_KIND_COUNT};
use crate::neural_style::{build_neural_style_weights, NUM_Z16_STYLES};
pub use crate::neural_style::z16_style_label;
use crate::palette::{palette_resolve_64, palette_tex_bytes_at};
use crate::surfel_gen::{SplatPlacement, SplatProto, SplatRef, SurfelProtoBank};
use crate::world_bridge::{AssembledFrame, VolSplat};
use crate::z16_world::{SurfaceVertex, Z16World, OCC_BRICK_SIZE, SN_BRICK_SIZE};
use bytemuck::{Pod, Zeroable};
use frostmarsh_proto::{WEATHER_BLIZZARD, WEATHER_EMBER, WEATHER_SQUALL};
use glam::Vec3;
use std::borrow::Cow;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    vp: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    origin_time: [f32; 4],
    viewport: [f32; 4],
    sun_dir: [f32; 4],
    sun_col: [f32; 4],
    sky_col: [f32; 4],
    bounce_col: [f32; 4],
    fog_col: [f32; 4], // w carries wind_x for volumetric drift
    point_pos: [[f32; 4]; 8],
    point_col: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveUniforms {
    low_res: [f32; 2],
    ss: f32,
    outline: f32,
    quantise: f32,
    dither: f32,
    crisp: f32,
    palette_count: f32,
    background: [f32; 4],
    palette: [[f32; 4]; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlitUniforms {
    art_res: [f32; 2],
    out_res: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudUniforms {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Z16Uniforms {
    inv_vp: [[f32; 4]; 4],
    vol_dims: [f32; 4],
    brick_dims: [f32; 4],
    world_offset: [f32; 4],
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    hero_world_pos: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StyleComputeParams {
    vol_dims: [u32; 4],
    region_min: [u32; 4],
    region_size: [u32; 4],
}

const STYLE_PARAMS_STRIDE: u64 = 256;
const MAX_STYLE_DIRTY_BRICKS: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct Atmosphere {
    pub weather_kind: u8,
    pub day_time: f32,
    pub wind_x: f32,
    pub wind_y: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            weather_kind: 0,
            day_time: 0.25,
            wind_x: 0.65,
            wind_y: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SmoothedAtmosphere {
    sun_col: [f32; 3],
    sky_col: [f32; 3],
    bounce_col: [f32; 3],
    fog_col: [f32; 3],
    exposure: f32,
    wind_drift: f32,
}

impl SmoothedAtmosphere {
    fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    }

    fn target_for(atmo: Atmosphere) -> Self {
        // Diurnal solar phase: day_time 0.0=midnight, 0.25=sunrise, 0.50=noon, 0.75=sunset
        let phase = atmo.day_time.rem_euclid(1.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        let solar = phase.sin(); // -1 midnight .. +1 noon
        let cos_p = phase.cos(); // +1 sunrise .. -1 sunset

        let day_lin = ((solar + 0.12) / 0.65).clamp(0.0, 1.0);
        let day_w = day_lin * day_lin * (3.0 - 2.0 * day_lin);
        let twilight_w = (1.0 - (solar.abs() / 0.45).clamp(0.0, 1.0)).powf(1.3) * 0.85;
        let dusk_f = ((-cos_p) * 0.5 + 0.5).clamp(0.0, 1.0);

        // 4 distinct Frostmarsh diurnal palettes:
        // Night (cool indigo moonlit), Noon (crisp bright), Dawn (golden amber), Dusk (warm crimson)
        let night_sun = [0.34, 0.48, 0.88];
        let night_sky = [0.10, 0.14, 0.30];
        let night_bnc = [0.08, 0.10, 0.22];
        let night_fog = [0.06, 0.08, 0.18];
        let night_exp = 0.86f32;

        let noon_sun = [1.14, 1.07, 0.96];
        let noon_sky = [0.38, 0.56, 0.82];
        let noon_bnc = [0.25, 0.30, 0.44];
        let noon_fog = [0.17, 0.24, 0.38];
        let noon_exp = 1.15f32;

        let dawn_sun = [1.28, 0.84, 0.46];
        let dawn_sky = [0.56, 0.42, 0.60];
        let dawn_bnc = [0.32, 0.23, 0.28];
        let dawn_fog = [0.28, 0.19, 0.27];
        let dawn_exp = 1.08f32;

        let dusk_sun = [1.32, 0.56, 0.32];
        let dusk_sky = [0.48, 0.23, 0.42];
        let dusk_bnc = [0.30, 0.15, 0.23];
        let dusk_fog = [0.24, 0.12, 0.21];
        let dusk_exp = 1.02f32;

        let tw_sun = Self::lerp3(dawn_sun, dusk_sun, dusk_f);
        let tw_sky = Self::lerp3(dawn_sky, dusk_sky, dusk_f);
        let tw_bnc = Self::lerp3(dawn_bnc, dusk_bnc, dusk_f);
        let tw_fog = Self::lerp3(dawn_fog, dusk_fog, dusk_f);
        let tw_exp = dawn_exp + (dusk_exp - dawn_exp) * dusk_f;

        let mut sun_col = Self::lerp3(Self::lerp3(night_sun, noon_sun, day_w), tw_sun, twilight_w);
        let mut sky_col = Self::lerp3(Self::lerp3(night_sky, noon_sky, day_w), tw_sky, twilight_w);
        let mut bounce_col = Self::lerp3(Self::lerp3(night_bnc, noon_bnc, day_w), tw_bnc, twilight_w);
        let mut fog_col = Self::lerp3(Self::lerp3(night_fog, noon_fog, day_w), tw_fog, twilight_w);
        let mut exposure = (night_exp + (noon_exp - night_exp) * day_w) * (1.0 - twilight_w) + tw_exp * twilight_w;

        // Modulate diurnal palette by live weather front (preserving day/night readability!)
        let wind_mul = match atmo.weather_kind {
            WEATHER_BLIZZARD => {
                let s = 0.45 + 0.55 * day_w;
                sun_col = Self::lerp3(sun_col, [0.76 * s, 0.90 * s, 1.12 * s], 0.50);
                sky_col = Self::lerp3(sky_col, [0.46 * s, 0.62 * s, 0.86 * s], 0.45);
                bounce_col = Self::lerp3(bounce_col, [0.28 * s, 0.36 * s, 0.52 * s], 0.45);
                fog_col = Self::lerp3(fog_col, [0.28 * s, 0.38 * s, 0.54 * s], 0.50);
                exposure *= 0.95;
                2.35
            }
            WEATHER_SQUALL => {
                let s = 0.45 + 0.55 * day_w;
                sun_col = Self::lerp3(sun_col, [0.78 * s, 0.82 * s, 0.94 * s], 0.45);
                sky_col = Self::lerp3(sky_col, [0.28 * s, 0.36 * s, 0.52 * s], 0.45);
                bounce_col = Self::lerp3(bounce_col, [0.18 * s, 0.22 * s, 0.32 * s], 0.45);
                fog_col = Self::lerp3(fog_col, [0.14 * s, 0.19 * s, 0.29 * s], 0.45);
                exposure *= 0.92;
                1.75
            }
            WEATHER_EMBER => {
                sun_col = Self::lerp3(sun_col, [1.24, 0.72, 0.42], 0.42);
                sky_col = Self::lerp3(sky_col, [0.46, 0.24, 0.38], 0.42);
                bounce_col = Self::lerp3(bounce_col, [0.30, 0.17, 0.22], 0.40);
                fog_col = Self::lerp3(fog_col, [0.24, 0.13, 0.20], 0.42);
                exposure *= 1.04;
                1.55
            }
            _ => 0.85,
        };
        let wind_mag = atmo.wind_x.hypot(atmo.wind_y).max(0.45);
        let sign = if atmo.wind_x < -0.05 { -1.0 } else { 1.0 };
        let wind_drift = sign * (wind_mag * 0.50 + 0.50) * wind_mul;
        Self {
            sun_col,
            sky_col,
            bounce_col,
            fog_col,
            exposure,
            wind_drift,
        }
    }

    fn lerp_towards(&mut self, target: Self, alpha: f32) {
        let a = alpha.clamp(0.0, 1.0);
        for i in 0..3 {
            self.sun_col[i] += (target.sun_col[i] - self.sun_col[i]) * a;
            self.sky_col[i] += (target.sky_col[i] - self.sky_col[i]) * a;
            self.bounce_col[i] += (target.bounce_col[i] - self.bounce_col[i]) * a;
            self.fog_col[i] += (target.fog_col[i] - self.fog_col[i]) * a;
        }
        self.exposure += (target.exposure - self.exposure) * a;
        self.wind_drift += (target.wind_drift - self.wind_drift) * a;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderStyle {
    pub label: &'static str,
    pub pixel_scale: u32,
    pub crisp_resolve: bool,
    pub outlines: bool,
    pub shadows: bool,
    pub dither: f32,
    pub cel_bands: f32,
}

impl RenderStyle {
    pub fn pixel_art(pixel_scale: u32) -> Self {
        Self {
            label: "PIXEL-2XSS",
            pixel_scale: pixel_scale.max(1),
            crisp_resolve: true,
            outlines: true,
            shadows: true,
            dither: 0.20,
            cel_bands: 9.0,
        }
    }

    pub fn clean_3d() -> Self {
        Self {
            label: "CLEAN-3D",
            pixel_scale: 1,
            crisp_resolve: false,
            outlines: false,
            shadows: true,
            dither: 0.0,
            cel_bands: 9.0,
        }
    }
}

pub struct Approach4Renderer {
    pub adapter_name: String,
    pub adapter_backend: String,
    pub out_format: wgpu::TextureFormat,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pipe_shadow: wgpu::RenderPipeline,
    pipe_mesh: wgpu::RenderPipeline,
    pipe_surfel: wgpu::RenderPipeline,
    pipe_vol: wgpu::RenderPipeline,
    pipe_resolve: wgpu::RenderPipeline,
    pipe_blit: wgpu::RenderPipeline,
    pipe_hud: wgpu::RenderPipeline,
    pipe_field_z16: wgpu::RenderPipeline,
    pipe_vol_z16: wgpu::RenderPipeline,
    pipe_neural_style_z16: wgpu::ComputePipeline,

    shadow_bgl: wgpu::BindGroupLayout,
    mesh_bgl: wgpu::BindGroupLayout,
    surfel_bgl: wgpu::BindGroupLayout,
    vol_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    hud_bgl: wgpu::BindGroupLayout,
    z16_mesh_bgl: wgpu::BindGroupLayout,
    z16_vol_bgl: wgpu::BindGroupLayout,
    z16_style_bgl: wgpu::BindGroupLayout,

    z16_active: bool,
    z16_style_mode: u32,
    z16_style_computed_for: Option<u32>,
    z16_style_full_dirty: bool,
    z16_style_dirty_bricks: Vec<(u32, u32, u32, u32)>,
    z16_hero_world_pos: [f32; 4],
    z16_dims: (u32, u32, u32),
    z16_occ_dims: (u32, u32, u32),
    z16_density_tex: Option<wgpu::Texture>,
    z16_density_view: Option<wgpu::TextureView>,
    z16_albedo_tex: Option<wgpu::Texture>,
    z16_albedo_view: Option<wgpu::TextureView>,
    z16_styled_albedo_tex: Option<wgpu::Texture>,
    z16_styled_albedo_view: Option<wgpu::TextureView>,
    z16_styled_albedo_storage_view: Option<wgpu::TextureView>,
    z16_occ_tex: Option<wgpu::Texture>,
    z16_occ_view: Option<wgpu::TextureView>,
    z16_vol_samp: wgpu::Sampler,
    z16_ubuf: wgpu::Buffer,
    z16_style_params_buf: wgpu::Buffer,
    z16_style_weights_buf: wgpu::Buffer,
    z16_vbuf: Option<wgpu::Buffer>,
    z16_vcap: usize,
    z16_ibuf: Option<wgpu::Buffer>,
    z16_icap: usize,
    z16_icount: u32,

    static_vbuf: wgpu::Buffer,
    static_ibuf: wgpu::Buffer,
    mesh_slices: [MeshSlice; MESH_KIND_COUNT],
    proto_buf: wgpu::Buffer,

    flash_pal_tex: wgpu::Texture,
    flash_pal_view: wgpu::TextureView,

    minimap_w: u32,
    minimap_h: u32,
    minimap_tex: wgpu::Texture,
    minimap_view: wgpu::TextureView,
    minimap_samp: wgpu::Sampler,

    shadow_view: wgpu::TextureView,
    shadow_samp: wgpu::Sampler,

    scene_ubuf: wgpu::Buffer,
    resolve_ubuf: wgpu::Buffer,
    blit_ubuf: wgpu::Buffer,
    hud_ubuf: wgpu::Buffer,
    identity_inst_buf: wgpu::Buffer,

    terrain_vbuf: wgpu::Buffer,
    terrain_vcap: usize,
    terrain_ibuf: wgpu::Buffer,
    terrain_icap: usize,
    mesh_inst_buf: wgpu::Buffer,
    mesh_inst_cap: usize,
    place_buf: wgpu::Buffer,
    place_cap: usize,
    ref_buf: wgpu::Buffer,
    ref_cap: usize,
    vol_buf: wgpu::Buffer,
    vol_cap: usize,
    hud_vbuf: wgpu::Buffer,
    hud_vcap: usize,

    ss_w: u32,
    ss_h: u32,
    ss_color_view: wgpu::TextureView,
    ss_depth_view: wgpu::TextureView,
    art_w: u32,
    art_h: u32,
    art_color_view: wgpu::TextureView,
    smooth_atmo: Option<SmoothedAtmosphere>,
    last_render_sim_time: f32,
}

fn make_2d_targets(
    device: &wgpu::Device,
    ss_w: u32,
    ss_h: u32,
    art_w: u32,
    art_h: u32,
) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
    let ss_col = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_ss_color"),
        size: wgpu::Extent3d { width: ss_w.max(1), height: ss_h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let ss_dep = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_ss_depth"),
        size: wgpu::Extent3d { width: ss_w.max(1), height: ss_h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let art_col = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_art_color"),
        size: wgpu::Extent3d { width: art_w.max(1), height: art_h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    (
        ss_col.create_view(&Default::default()),
        ss_dep.create_view(&Default::default()),
        art_col.create_view(&Default::default()),
    )
}

fn make_minimap_tex(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tactical_minimap_tex"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

impl Approach4Renderer {
    pub async fn new_headless(
        meshes: &StaticMeshBank,
        surfels: &SurfelProtoBank,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await?;
        Self::from_adapter(adapter, wgpu::TextureFormat::Rgba8Unorm, meshes, surfels).await
    }

    pub async fn from_adapter(
        adapter: wgpu::Adapter,
        out_format: wgpu::TextureFormat,
        meshes: &StaticMeshBank,
        surfels: &SurfelProtoBank,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let adapter_backend = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("frostmarsh_unified_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await?;

        let sh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_pass"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/shadow_pass.wgsl"
            ))),
        });
        let mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_pass"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/mesh_pass.wgsl"
            ))),
        });
        let surfel_src = include_str!("../assets/shaders/surfel_instanced.wgsl").replace(
            "let sway = sin(t * 2.2 + pl.lod_phase.y + rel_pos.x * 0.35 + rel_pos.z * 0.25) * 0.14 * h_factor;",
            "let wind_mul = clamp(0.60 + abs(u.fog_col.w) * 0.85, 0.60, 2.80);\n        let sway = sin(t * (1.8 + wind_mul * 0.9) + pl.lod_phase.y + rel_pos.x * 0.35 + rel_pos.z * 0.25) * 0.14 * wind_mul * h_factor + u.fog_col.w * 0.05 * h_factor;",
        );
        let surfel_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surfel_instanced"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(surfel_src)),
        });
        let vol_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_splat"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/volumetric_splat.wgsl"
            ))),
        });
        let resolve_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel_resolve"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/pixel_resolve.wgsl"
            ))),
        });
        let blit_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit_pass"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/blit_pass.wgsl"
            ))),
        });
        let hud_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud_pass"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/hud_pass.wgsl"
            ))),
        });
        let field_z16_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_mesh_z16"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/field_mesh_z16.wgsl"
            ))),
        });
        let vol_z16_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_overlay_z16"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/volumetric_overlay_z16.wgsl"
            ))),
        });
        let neural_style_z16_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neural_style_z16"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/neural_style_z16.wgsl"
            ))),
        });

        let static_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("static_vbuf"),
            size: (meshes.vertices.len() * std::mem::size_of::<MeshVertex>()).max(64) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&static_vbuf, 0, bytemuck::cast_slice(&meshes.vertices));

        let static_ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("static_ibuf"),
            size: (meshes.indices.len() * 4).max(64) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&static_ibuf, 0, bytemuck::cast_slice(&meshes.indices));

        let proto_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surfel_protos"),
            size: (surfels.protos.len() * std::mem::size_of::<SplatProto>()).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&proto_buf, 0, bytemuck::cast_slice(&surfels.protos));

        // 1 KB (256×1 RGBA8) animated `flash()` palette
        let flash_pal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("flash_pal_1kb"),
            size: wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let init_pal = palette_tex_bytes_at(0.0);
        queue.write_texture(
            flash_pal_tex.as_image_copy(),
            &init_pal,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
        let flash_pal_view = flash_pal_tex.create_view(&Default::default());

        let (minimap_tex, minimap_view) = make_minimap_tex(&device, 256, 256);
        let minimap_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("minimap_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map_2048"),
            size: wgpu::Extent3d { width: 2048, height: 2048, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let shadow_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_cmp_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let mesh_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 40, shader_location: 3 },
            ],
        };

        let hud_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<HudVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 32, shader_location: 3 },
            ],
        };

        let ubuf_entry = |binding, vis| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sbuf_ro_entry = |binding, vis| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX),
                sbuf_ro_entry(1, wgpu::ShaderStages::VERTEX),
            ],
        });

        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                sbuf_ro_entry(1, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let surfel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfel_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                sbuf_ro_entry(1, wgpu::ShaderStages::VERTEX),
                sbuf_ro_entry(2, wgpu::ShaderStages::VERTEX),
                sbuf_ro_entry(3, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let vol_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                sbuf_ro_entry(1, wgpu::ShaderStages::VERTEX),
            ],
        });

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let hud_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let z16_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("z16_mesh_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                ubuf_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let z16_vol_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("z16_vol_bgl"),
            entries: &[
                ubuf_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                ubuf_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let z16_style_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("z16_style_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<StyleComputeParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let depth_opaque = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        });

        let depth_vol_test_only = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        });

        let pipe_shadow = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_shadow"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&shadow_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &sh_mod,
                entry_point: Some("vs_mesh_shadow"),
                buffers: &[Some(mesh_vbl.clone())],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_mesh = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_mesh"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&mesh_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &mesh_mod,
                entry_point: Some("vs_main"),
                buffers: &[Some(mesh_vbl)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: depth_opaque.clone(),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_surfel = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_surfel"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&surfel_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &surfel_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &surfel_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_opaque,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_vol = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_volumetric_3dgs"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&vol_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &vol_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &vol_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_vol_test_only,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_resolve"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&resolve_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &resolve_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_blit"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&blit_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &blit_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: out_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_hud = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_hud"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&hud_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &hud_mod,
                entry_point: Some("vs_main"),
                buffers: &[Some(hud_vbl)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: out_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let sn_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SurfaceVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };

        let pipe_field_z16 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_field_z16"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&z16_mesh_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &field_z16_mod,
                entry_point: Some("vs_main"),
                buffers: &[Some(sn_vbl)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &field_z16_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_vol_z16 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_vol_z16"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&z16_vol_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &vol_z16_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &vol_z16_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_neural_style_z16 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipe_neural_style_z16"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("z16_style_pl"),
                bind_group_layouts: &[Some(&z16_style_bgl)],
                immediate_size: 0,
            })),
            module: &neural_style_z16_mod,
            entry_point: Some("cs_style_z16"),
            compilation_options: Default::default(),
            cache: None,
        });

        let z16_style_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("z16_style_params_ubuf"),
            size: STYLE_PARAMS_STRIDE * MAX_STYLE_DIRTY_BRICKS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let style_weights = build_neural_style_weights();
        let z16_style_weights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("z16_neural_style_weights_4kb_x4"),
            size: (style_weights.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&z16_style_weights_buf, 0, bytemuck::cast_slice(&style_weights));

        let z16_vol_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("z16_vol_trilinear_samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let mk_ubuf = |sz: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: sz as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let scene_ubuf = mk_ubuf(std::mem::size_of::<SceneUniforms>());
        let resolve_ubuf = mk_ubuf(std::mem::size_of::<ResolveUniforms>());
        let blit_ubuf = mk_ubuf(std::mem::size_of::<BlitUniforms>());
        let hud_ubuf = mk_ubuf(std::mem::size_of::<HudUniforms>());
        let z16_ubuf = mk_ubuf(std::mem::size_of::<Z16Uniforms>());

        let identity = MeshInstance {
            pos_yaw: [0.0, 0.0, 0.0, 0.0],
            scale_pitch: [1.0, 1.0, 1.0, 0.0],
            tint_glow: [1.0, 1.0, 1.0, 0.0],
            flags: [-1.0, 0.0, 0.0, 0.0],
        };
        let identity_inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<MeshInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&identity_inst_buf, 0, bytemuck::bytes_of(&identity));

        let mk_buf = |bytes: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes.max(256) as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let terrain_vcap = 131_072;
        let terrain_icap = 524_288;
        let mesh_inst_cap = 32_768;
        let place_cap = 16_384;
        let ref_cap = 524_288;
        let vol_cap = 16_384;
        let hud_vcap = 65_536;

        let terrain_vbuf = mk_buf(terrain_vcap * std::mem::size_of::<MeshVertex>(), wgpu::BufferUsages::VERTEX);
        let terrain_ibuf = mk_buf(terrain_icap * 4, wgpu::BufferUsages::INDEX);
        let mesh_inst_buf = mk_buf(mesh_inst_cap * std::mem::size_of::<MeshInstance>(), wgpu::BufferUsages::STORAGE);
        let place_buf = mk_buf(place_cap * std::mem::size_of::<SplatPlacement>(), wgpu::BufferUsages::STORAGE);
        let ref_buf = mk_buf(ref_cap * std::mem::size_of::<SplatRef>(), wgpu::BufferUsages::STORAGE);
        let vol_buf = mk_buf(vol_cap * std::mem::size_of::<VolSplat>(), wgpu::BufferUsages::STORAGE);
        let hud_vbuf = mk_buf(hud_vcap * std::mem::size_of::<HudVertex>(), wgpu::BufferUsages::VERTEX);

        let (ss_cv, ss_dv, art_cv) = make_2d_targets(&device, 1920, 1080, 960, 540);

        Ok(Self {
            adapter_name,
            adapter_backend,
            out_format,
            device,
            queue,
            pipe_shadow,
            pipe_mesh,
            pipe_surfel,
            pipe_vol,
            pipe_resolve,
            pipe_blit,
            pipe_hud,
            pipe_field_z16,
            pipe_vol_z16,
            pipe_neural_style_z16,
            shadow_bgl,
            mesh_bgl,
            surfel_bgl,
            vol_bgl,
            resolve_bgl,
            blit_bgl,
            hud_bgl,
            z16_mesh_bgl,
            z16_vol_bgl,
            z16_style_bgl,
            z16_active: false,
            z16_style_mode: 0,
            z16_style_computed_for: None,
            z16_style_full_dirty: true,
            z16_style_dirty_bricks: Vec::new(),
            z16_hero_world_pos: [0.0; 4],
            z16_dims: (0, 0, 0),
            z16_occ_dims: (0, 0, 0),
            z16_density_tex: None,
            z16_density_view: None,
            z16_albedo_tex: None,
            z16_albedo_view: None,
            z16_styled_albedo_tex: None,
            z16_styled_albedo_view: None,
            z16_styled_albedo_storage_view: None,
            z16_occ_tex: None,
            z16_occ_view: None,
            z16_vol_samp,
            z16_ubuf,
            z16_style_params_buf,
            z16_style_weights_buf,
            z16_vbuf: None,
            z16_vcap: 0,
            z16_ibuf: None,
            z16_icap: 0,
            z16_icount: 0,
            static_vbuf,
            static_ibuf,
            mesh_slices: meshes.slices,
            proto_buf,
            flash_pal_tex,
            flash_pal_view,
            minimap_w: 256,
            minimap_h: 256,
            minimap_tex,
            minimap_view,
            minimap_samp,
            shadow_view,
            shadow_samp,
            scene_ubuf,
            resolve_ubuf,
            blit_ubuf,
            hud_ubuf,
            identity_inst_buf,
            terrain_vbuf,
            terrain_vcap,
            terrain_ibuf,
            terrain_icap,
            mesh_inst_buf,
            mesh_inst_cap,
            place_buf,
            place_cap,
            ref_buf,
            ref_cap,
            vol_buf,
            vol_cap,
            hud_vbuf,
            hud_vcap,
            ss_w: 1920,
            ss_h: 1080,
            ss_color_view: ss_cv,
            ss_depth_view: ss_dv,
            art_w: 960,
            art_h: 540,
            art_color_view: art_cv,
            smooth_atmo: None,
            last_render_sim_time: 0.0,
        })
    }

    pub fn sun_dir(atmo: Atmosphere) -> Vec3 {
        let phase = atmo.day_time.rem_euclid(1.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        let solar = phase.sin();
        let horiz = phase.cos();
        let sun_v = Vec3::new(
            -horiz * 0.85,
            0.24 + 0.66 * solar.clamp(0.0, 1.0).powf(0.75),
            -0.38 - 0.18 * solar.clamp(0.0, 1.0),
        )
        .normalize();
        let moon_v = Vec3::new(
            horiz * 0.62,
            0.26 + 0.42 * (-solar).clamp(0.0, 1.0),
            -0.46,
        )
        .normalize();
        let blend = ((solar + 0.15) / 0.20).clamp(0.0, 1.0);
        let s_blend = blend * blend * (3.0 - 2.0 * blend);
        moon_v.lerp(sun_v, s_blend).normalize()
    }

    pub fn set_z16_mode(&mut self, active: bool, style_mode: u32) {
        self.z16_active = active;
        self.z16_style_mode = style_mode % NUM_Z16_STYLES;
    }

    /// Sets the local player's 3D world position and cutaway strength (`0.0` disabled .. `1.0` active)
    /// for the Smart 3D Cave & Under-Bridge Ceiling Cutaway in `field_mesh_z16.wgsl`.
    pub fn set_z16_hero_cutaway(&mut self, world_pos: Vec3, enabled: f32) {
        self.z16_hero_world_pos = [
            world_pos.x,
            world_pos.y,
            world_pos.z,
            enabled.clamp(0.0, 1.0),
        ];
    }

    /// Lazily allocate/resize the 3D `Rgba8Unorm` GPU textures (`z16_density_tex`, `z16_albedo_tex`,
    /// `z16_occ_tex`) and Surface-Nets vertex/index buffers, uploading only dirty `16×16×D` bricks!
    pub fn upload_z16_world(&mut self, z16: &mut Z16World) {
        let dims = (z16.width as u32, z16.depth as u32, z16.height as u32);
        let occ_dims = (z16.occ_bx as u32, z16.occ_by as u32, z16.occ_bz as u32);
        if dims.0 == 0 || dims.1 == 0 || dims.2 == 0 {
            return;
        }

        let mk_d3 = |device: &wgpu::Device, label: &str, (w, d, h): (u32, u32, u32)| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: d,
                    depth_or_array_layers: h,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });
            (tex, view)
        };

        if self.z16_dims != dims || self.z16_density_tex.is_none() || self.z16_styled_albedo_tex.is_none() {
            let (dt, dv) = mk_d3(&self.device, "z16_density_tex", dims);
            let (at, av) = mk_d3(&self.device, "z16_albedo_tex", dims);
            let styled_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("z16_styled_albedo_tex"),
                size: wgpu::Extent3d {
                    width: dims.0,
                    height: dims.1,
                    depth_or_array_layers: dims.2,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let styled_view = styled_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });
            let styled_storage_view = styled_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });
            self.z16_density_tex = Some(dt);
            self.z16_density_view = Some(dv);
            self.z16_albedo_tex = Some(at);
            self.z16_albedo_view = Some(av);
            self.z16_styled_albedo_tex = Some(styled_tex);
            self.z16_styled_albedo_view = Some(styled_view);
            self.z16_styled_albedo_storage_view = Some(styled_storage_view);
            self.z16_dims = dims;
            self.z16_style_computed_for = None;
            self.z16_style_full_dirty = true;
            self.z16_style_dirty_bricks.clear();
            z16.gpu_tex_full_dirty = true;
        }
        if self.z16_occ_dims != occ_dims || self.z16_occ_tex.is_none() {
            let (ot, ov) = mk_d3(&self.device, "z16_occ_tex", occ_dims);
            self.z16_occ_tex = Some(ot);
            self.z16_occ_view = Some(ov);
            self.z16_occ_dims = occ_dims;
            z16.gpu_occ_dirty = true;
        }

        // Upload 3D density & albedo textures (full or sub-brick incremental)
        if z16.gpu_tex_full_dirty {
            let sz = wgpu::Extent3d {
                width: dims.0,
                height: dims.1,
                depth_or_array_layers: dims.2,
            };
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims.0 * 4),
                rows_per_image: Some(dims.1),
            };
            if let Some(ref dt) = self.z16_density_tex {
                self.queue.write_texture(dt.as_image_copy(), &z16.density_mat, layout, sz);
            }
            if let Some(ref at) = self.z16_albedo_tex {
                self.queue.write_texture(at.as_image_copy(), &z16.albedo_thermal, layout, sz);
            }
            z16.gpu_tex_full_dirty = false;
            z16.gpu_dirty_bricks.clear();
            self.z16_style_full_dirty = true;
            self.z16_style_dirty_bricks.clear();
        } else if !z16.gpu_dirty_bricks.is_empty() {
            let dirty = std::mem::take(&mut z16.gpu_dirty_bricks);
            let w = z16.width;
            let d = z16.depth;
            let h = z16.height;
            let mut tmp_dm = Vec::with_capacity(SN_BRICK_SIZE * d * SN_BRICK_SIZE * 4);
            let mut tmp_at = Vec::with_capacity(SN_BRICK_SIZE * d * SN_BRICK_SIZE * 4);
            for (bx, bz) in dirty {
                let x0 = bx * SN_BRICK_SIZE;
                let z0 = bz * SN_BRICK_SIZE;
                let cw = SN_BRICK_SIZE.min(w.saturating_sub(x0));
                let ch = SN_BRICK_SIZE.min(h.saturating_sub(z0));
                if cw == 0 || ch == 0 {
                    continue;
                }
                if !self.z16_style_full_dirty {
                    if self.z16_style_dirty_bricks.len() < MAX_STYLE_DIRTY_BRICKS {
                        self.z16_style_dirty_bricks.push((x0 as u32, z0 as u32, cw as u32, ch as u32));
                    } else {
                        self.z16_style_full_dirty = true;
                        self.z16_style_dirty_bricks.clear();
                    }
                }
                let row_bytes = cw * 4;
                tmp_dm.clear();
                tmp_at.clear();
                for dz in 0..ch {
                    let iz = z0 + dz;
                    for iy in 0..d {
                        let off = ((iz * d + iy) * w + x0) * 4;
                        tmp_dm.extend_from_slice(&z16.density_mat[off..off + row_bytes]);
                        tmp_at.extend_from_slice(&z16.albedo_thermal[off..off + row_bytes]);
                    }
                }
                let origin = wgpu::Origin3d {
                    x: x0 as u32,
                    y: 0,
                    z: z0 as u32,
                };
                let sz = wgpu::Extent3d {
                    width: cw as u32,
                    height: d as u32,
                    depth_or_array_layers: ch as u32,
                };
                let layout = wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((cw * 4) as u32),
                    rows_per_image: Some(d as u32),
                };
                if let Some(ref dt) = self.z16_density_tex {
                    self.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: dt,
                            mip_level: 0,
                            origin,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &tmp_dm,
                        layout,
                        sz,
                    );
                }
                if let Some(ref at) = self.z16_albedo_tex {
                    self.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: at,
                            mip_level: 0,
                            origin,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &tmp_at,
                        layout,
                        sz,
                    );
                }
            }
        }

        // Upload 8×8×8 brick occupancy 3D texture
        if z16.gpu_occ_dirty {
            if let Some(ref ot) = self.z16_occ_tex {
                self.queue.write_texture(
                    ot.as_image_copy(),
                    &z16.occ_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(occ_dims.0 * 4),
                        rows_per_image: Some(occ_dims.1),
                    },
                    wgpu::Extent3d {
                        width: occ_dims.0,
                        height: occ_dims.1,
                        depth_or_array_layers: occ_dims.2,
                    },
                );
            }
            z16.gpu_occ_dirty = false;
        }

        // Upload Surface-Nets vertex & index buffers
        if z16.gpu_mesh_dirty {
            let vlen = z16.flat_verts.len();
            let ilen = z16.flat_idxs.len();
            if vlen > self.z16_vcap || self.z16_vbuf.is_none() {
                self.z16_vcap = (vlen * 2).max(4096);
                self.z16_vbuf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("z16_sn_vbuf"),
                    size: (self.z16_vcap * std::mem::size_of::<SurfaceVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            if ilen > self.z16_icap || self.z16_ibuf.is_none() {
                self.z16_icap = (ilen * 2).max(16_384);
                self.z16_ibuf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("z16_sn_ibuf"),
                    size: (self.z16_icap * 4) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            if vlen > 0 {
                if let Some(ref vb) = self.z16_vbuf {
                    self.queue.write_buffer(vb, 0, bytemuck::cast_slice(&z16.flat_verts));
                }
            }
            if ilen > 0 {
                if let Some(ref ib) = self.z16_ibuf {
                    self.queue.write_buffer(ib, 0, bytemuck::cast_slice(&z16.flat_idxs));
                }
            }
            self.z16_icount = ilen as u32;
            z16.gpu_mesh_dirty = false;
        }
    }

    pub fn upload_minimap(&mut self, w: u32, h: u32, rgba: &[u8]) {
        if w == 0 || h == 0 || rgba.len() < (w * h * 4) as usize {
            return;
        }
        if self.minimap_w != w || self.minimap_h != h {
            let (t, v) = make_minimap_tex(&self.device, w, h);
            self.minimap_tex = t;
            self.minimap_view = v;
            self.minimap_w = w;
            self.minimap_h = h;
        }
        self.queue.write_texture(
            self.minimap_tex.as_image_copy(),
            &rgba[..(w * h * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    pub fn create_output_texture(&self, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless_out"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.out_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_into(
        &mut self,
        dst_view: &wgpu::TextureView,
        out_w: u32,
        out_h: u32,
        style: &RenderStyle,
        atmo: Atmosphere,
        camera: &MobaCamera,
        mats: &CameraMatrices,
        sim_time: f32,
        frame: &AssembledFrame,
        hud: Option<&HudBatch>,
    ) -> (u32, u32) {
        let px = style.pixel_scale.max(1);
        let art_w = (out_w / px).max(64);
        let art_h = (out_h / px).max(36);
        let ss: u32 = if style.crisp_resolve && px >= 2 { 2 } else { 1 };
        let ss_w = art_w * ss;
        let ss_h = art_h * ss;

        if self.ss_w != ss_w || self.ss_h != ss_h || self.art_w != art_w || self.art_h != art_h {
            let (sc, sd, ac) = make_2d_targets(&self.device, ss_w, ss_h, art_w, art_h);
            self.ss_w = ss_w;
            self.ss_h = ss_h;
            self.art_w = art_w;
            self.art_h = art_h;
            self.ss_color_view = sc;
            self.ss_depth_view = sd;
            self.art_color_view = ac;
        }

        if frame.terrain_verts.len() > self.terrain_vcap {
            self.terrain_vcap = (frame.terrain_verts.len() * 2).max(1024);
            self.terrain_vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (self.terrain_vcap * std::mem::size_of::<MeshVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if frame.terrain_idxs.len() > self.terrain_icap {
            self.terrain_icap = (frame.terrain_idxs.len() * 2).max(1024);
            self.terrain_ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (self.terrain_icap * 4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let mut flat_meshes: Vec<MeshInstance> = Vec::new();
        let mut batch_ranges: [(u32, u32); MESH_KIND_COUNT] = [(0, 0); MESH_KIND_COUNT];
        for (k, batch) in frame.mesh_batches.iter().enumerate() {
            let start = flat_meshes.len() as u32;
            flat_meshes.extend_from_slice(batch);
            batch_ranges[k] = (start, flat_meshes.len() as u32);
        }
        if flat_meshes.len() > self.mesh_inst_cap {
            self.mesh_inst_cap = (flat_meshes.len() * 2).max(1024);
            self.mesh_inst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (self.mesh_inst_cap * std::mem::size_of::<MeshInstance>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if frame.surfel_placements.len() > self.place_cap {
            self.place_cap = (frame.surfel_placements.len() * 2).max(256);
            self.place_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (self.place_cap * std::mem::size_of::<SplatPlacement>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if frame.surfel_refs.len() > self.ref_cap {
            self.ref_cap = (frame.surfel_refs.len() * 2).max(1024);
            self.ref_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (self.ref_cap * std::mem::size_of::<SplatRef>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if frame.vol_splats.len() > self.vol_cap {
            self.vol_cap = (frame.vol_splats.len() * 2).max(256);
            self.vol_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (self.vol_cap * std::mem::size_of::<VolSplat>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if !frame.terrain_verts.is_empty() {
            self.queue.write_buffer(&self.terrain_vbuf, 0, bytemuck::cast_slice(&frame.terrain_verts));
        }
        if !frame.terrain_idxs.is_empty() {
            self.queue.write_buffer(&self.terrain_ibuf, 0, bytemuck::cast_slice(&frame.terrain_idxs));
        }
        if !flat_meshes.is_empty() {
            self.queue.write_buffer(&self.mesh_inst_buf, 0, bytemuck::cast_slice(&flat_meshes));
        }
        if !frame.surfel_placements.is_empty() {
            self.queue.write_buffer(&self.place_buf, 0, bytemuck::cast_slice(&frame.surfel_placements));
        }
        if !frame.surfel_refs.is_empty() {
            self.queue.write_buffer(&self.ref_buf, 0, bytemuck::cast_slice(&frame.surfel_refs));
        }
        if !frame.vol_splats.is_empty() {
            self.queue.write_buffer(&self.vol_buf, 0, bytemuck::cast_slice(&frame.vol_splats));
        }

        let hud_vert_count = if let Some(hb) = hud {
            if hb.vertices.len() > self.hud_vcap {
                self.hud_vcap = (hb.vertices.len() * 2).max(1024);
                self.hud_vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (self.hud_vcap * std::mem::size_of::<HudVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if !hb.vertices.is_empty() {
                self.queue.write_buffer(&self.hud_vbuf, 0, bytemuck::cast_slice(&hb.vertices));
            }
            hb.vertices.len() as u32
        } else {
            0
        };

        // 1 KB animated `flash()` palette
        let pal_bytes = palette_tex_bytes_at(sim_time);
        self.queue.write_texture(
            self.flash_pal_tex.as_image_copy(),
            &pal_bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );

        let sun_dir = Self::sun_dir(atmo);
        let target_atmo = SmoothedAtmosphere::target_for(atmo);
        let dt_atmo = (sim_time - self.last_render_sim_time).clamp(0.0, 0.25);
        self.last_render_sim_time = sim_time;
        let cur_atmo = match self.smooth_atmo.as_mut() {
            Some(cur) => {
                let alpha = 1.0 - (-dt_atmo * 2.4).exp();
                cur.lerp_towards(target_atmo, alpha);
                *cur
            }
            None => {
                self.smooth_atmo = Some(target_atmo);
                target_atmo
            }
        };
        let mut sun_col = cur_atmo.sun_col;
        let mut sky_col = cur_atmo.sky_col;
        let mut bounce_col = cur_atmo.bounce_col;
        let mut fog_col = cur_atmo.fog_col;

        // Style-aware sky/atmosphere harmonization when live Z=16..32 client WGSL neural style swap (`Y`-key) is active!
        let active_style_id = self.z16_style_mode % NUM_Z16_STYLES;
        let (z16_sky_top, z16_sky_hor, z16_bg_srgb) = if self.z16_active {
            match active_style_id {
                1 => {
                    // Style 1: VOLCANIC EMBER — blazing fiery amber sun, smoky crimson-charcoal sky
                    sun_col = [1.45, 0.72, 0.34];
                    sky_col = [0.46, 0.18, 0.16];
                    bounce_col = [0.34, 0.13, 0.10];
                    fog_col = [0.22, 0.09, 0.08];
                    ([0.22, 0.07, 0.07], [0.48, 0.19, 0.11], [0.22, 0.09, 0.08])
                }
                2 => {
                    // Style 2: ARCANE THERMAL — cool electric-cyan moonlight, deep midnight-indigo sky
                    sun_col = [0.68, 0.94, 1.32];
                    sky_col = [0.14, 0.28, 0.56];
                    bounce_col = [0.12, 0.14, 0.36];
                    fog_col = [0.06, 0.10, 0.22];
                    ([0.04, 0.09, 0.28], [0.11, 0.24, 0.44], [0.06, 0.10, 0.22])
                }
                3 => {
                    // Style 3: EMERALD CAVERN — luminous jade mist, deep emerald-teal twilight sky
                    sun_col = [0.82, 1.32, 0.96];
                    sky_col = [0.12, 0.44, 0.36];
                    bounce_col = [0.09, 0.28, 0.22];
                    fog_col = [0.07, 0.18, 0.15];
                    ([0.04, 0.20, 0.16], [0.12, 0.38, 0.30], [0.07, 0.18, 0.15])
                }
                _ => {
                    // Style 0: NORDIC — crisp alpine sky modulated by live diurnal/weather atmosphere
                    let top = [
                        (sky_col[0] * 0.62).powf(1.6),
                        (sky_col[1] * 0.88).powf(1.5),
                        (sky_col[2] * 1.15).powf(1.4),
                    ];
                    let hor = [
                        (fog_col[0] * 1.35 + 0.12).powf(1.6),
                        (fog_col[1] * 1.45 + 0.16).powf(1.5),
                        (fog_col[2] * 1.55 + 0.20).powf(1.4),
                    ];
                    (top, hor, fog_col)
                }
            }
        } else {
            ([0.14, 0.30, 0.62], [0.42, 0.56, 0.76], fog_col)
        };

        let scene_u = SceneUniforms {
            view: mats.view.to_cols_array_2d(),
            proj: mats.proj.to_cols_array_2d(),
            vp: mats.vp.to_cols_array_2d(),
            light_vp: mats.light_vp.to_cols_array_2d(),
            cam_pos: [mats.rel_eye.x, mats.rel_eye.y, mats.rel_eye.z, mats.focal_px],
            cam_fwd: [mats.fwd.x, mats.fwd.y, mats.fwd.z, if camera.ortho { 1.0 } else { 0.0 }],
            origin_time: [mats.origin.x, mats.origin.y, mats.origin.z, sim_time],
            viewport: [ss_w as f32, ss_h as f32, style.cel_bands, 0.0012],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, cur_atmo.exposure],
            sun_col: [sun_col[0], sun_col[1], sun_col[2], if style.shadows { 0.72 } else { 0.0 }],
            sky_col: [sky_col[0], sky_col[1], sky_col[2], 120.0],
            bounce_col: [bounce_col[0], bounce_col[1], bounce_col[2], 280.0],
            fog_col: [fog_col[0], fog_col[1], fog_col[2], cur_atmo.wind_drift],
            point_pos: frame.point_pos,
            point_col: frame.point_col,
        };
        self.queue.write_buffer(&self.scene_ubuf, 0, bytemuck::bytes_of(&scene_u));

        let z16_ready = self.z16_active
            && self.z16_density_view.is_some()
            && self.z16_albedo_view.is_some()
            && self.z16_styled_albedo_view.is_some()
            && self.z16_styled_albedo_storage_view.is_some()
            && self.z16_occ_view.is_some();

        if z16_ready {
            let z16_u = Z16Uniforms {
                inv_vp: mats.vp.inverse().to_cols_array_2d(),
                vol_dims: [
                    self.z16_dims.0 as f32,
                    self.z16_dims.1 as f32,
                    self.z16_dims.2 as f32,
                    active_style_id as f32,
                ],
                brick_dims: [
                    self.z16_occ_dims.0 as f32,
                    self.z16_occ_dims.1 as f32,
                    self.z16_occ_dims.2 as f32,
                    OCC_BRICK_SIZE as f32,
                ],
                world_offset: [
                    self.z16_dims.0 as f32 * 0.5,
                    0.0,
                    self.z16_dims.2 as f32 * 0.5,
                    if style.crisp_resolve { 1.0 } else { 0.0 },
                ],
                sky_top: [z16_sky_top[0], z16_sky_top[1], z16_sky_top[2], sim_time],
                sky_horizon: [z16_sky_hor[0], z16_sky_hor[1], z16_sky_hor[2], cur_atmo.exposure * 1.05],
                hero_world_pos: self.z16_hero_world_pos,
            };
            self.queue.write_buffer(&self.z16_ubuf, 0, bytemuck::bytes_of(&z16_u));
        }

        let (pal_arr, pal_n) = palette_resolve_64();
        let resolve_u = ResolveUniforms {
            low_res: [art_w as f32, art_h as f32],
            ss: ss as f32,
            outline: if style.outlines { 1.0 } else { 0.0 },
            quantise: if style.crisp_resolve { 1.0 } else { 0.0 },
            dither: style.dither,
            crisp: if style.crisp_resolve { 1.0 } else { 0.0 },
            palette_count: pal_n,
            background: [z16_bg_srgb[0], z16_bg_srgb[1], z16_bg_srgb[2], 1.0],
            palette: pal_arr,
        };
        self.queue.write_buffer(&self.resolve_ubuf, 0, bytemuck::bytes_of(&resolve_u));

        let blit_u = BlitUniforms {
            art_res: [art_w as f32, art_h as f32],
            out_res: [out_w as f32, out_h as f32],
        };
        self.queue.write_buffer(&self.blit_ubuf, 0, bytemuck::bytes_of(&blit_u));

        let hud_u = HudUniforms {
            screen_size: [out_w as f32, out_h as f32],
            _pad: [0.0, 0.0],
        };
        self.queue.write_buffer(&self.hud_ubuf, 0, bytemuck::bytes_of(&hud_u));

        let shadow_terrain_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.identity_inst_buf.as_entire_binding() },
            ],
        });
        let shadow_mesh_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.mesh_inst_buf.as_entire_binding() },
            ],
        });
        let mesh_terrain_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.identity_inst_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.shadow_samp) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.flash_pal_view) },
            ],
        });
        let mesh_props_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.mesh_inst_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.shadow_samp) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.flash_pal_view) },
            ],
        });
        let surfel_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.surfel_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.proto_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.place_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.ref_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.shadow_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.shadow_samp) },
            ],
        });
        let vol_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.vol_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vol_buf.as_entire_binding() },
            ],
        });
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.resolve_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ss_color_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.ss_depth_view) },
            ],
        });
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.blit_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.art_color_view) },
            ],
        });
        let hud_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.hud_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.hud_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.minimap_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.minimap_samp) },
            ],
        });

        let z16_mesh_bg = if z16_ready {
            Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("z16_mesh_bg"),
                layout: &self.z16_mesh_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.z16_ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(self.z16_density_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(self.z16_styled_albedo_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(self.z16_occ_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.z16_vol_samp),
                    },
                ],
            }))
        } else {
            None
        };
        let z16_vol_bg = if z16_ready {
            Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("z16_vol_bg"),
                layout: &self.z16_vol_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.z16_ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(self.z16_density_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(self.z16_styled_albedo_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(self.z16_occ_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.z16_vol_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.ss_depth_view),
                    },
                ],
            }))
        } else {
            None
        };

        let mut enc = self.device.create_command_encoder(&Default::default());

        // PASS 0 (Z=16..32): Live Client-Side WGSL Neural Style Compute (`neural_style_z16.wgsl`)
        // Runs ONLY on dirty bricks OR instantaneously on `Y`-key style swap (0 ms overhead on steady frames!).
        if z16_ready {
            let need_full = self.z16_style_full_dirty
                || self.z16_style_computed_for != Some(active_style_id);
            let dirty_bricks = if need_full {
                self.z16_style_dirty_bricks.clear();
                Vec::new()
            } else {
                std::mem::take(&mut self.z16_style_dirty_bricks)
            };

            if need_full || !dirty_bricks.is_empty() {
                let style_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("z16_style_bg"),
                    layout: &self.z16_style_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.z16_style_params_buf,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<StyleComputeParams>() as u64,
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                self.z16_density_view.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                self.z16_albedo_view.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.z16_style_weights_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                self.z16_styled_albedo_storage_view.as_ref().unwrap(),
                            ),
                        },
                    ],
                });

                let (vw, vd, vh) = self.z16_dims;
                if need_full {
                    let p = StyleComputeParams {
                        vol_dims: [vw, vd, vh, active_style_id],
                        region_min: [0, 0, 0, 0],
                        region_size: [vw, vd, vh, 0],
                    };
                    self.queue.write_buffer(&self.z16_style_params_buf, 0, bytemuck::bytes_of(&p));
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("neural_style_z16_full_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&self.pipe_neural_style_z16);
                    cp.set_bind_group(0, &style_bg, &[0]);
                    cp.dispatch_workgroups(vw.div_ceil(8), vd.div_ceil(4), vh.div_ceil(8));
                    self.z16_style_full_dirty = false;
                    self.z16_style_computed_for = Some(active_style_id);
                } else {
                    for (i, &(x0, z0, cw, ch)) in dirty_bricks.iter().enumerate() {
                        let p = StyleComputeParams {
                            vol_dims: [vw, vd, vh, active_style_id],
                            region_min: [x0, 0, z0, 0],
                            region_size: [cw, vd, ch, 0],
                        };
                        self.queue.write_buffer(
                            &self.z16_style_params_buf,
                            i as u64 * STYLE_PARAMS_STRIDE,
                            bytemuck::bytes_of(&p),
                        );
                    }
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("neural_style_z16_bricks_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&self.pipe_neural_style_z16);
                    for (i, &(_x0, _z0, cw, ch)) in dirty_bricks.iter().enumerate() {
                        cp.set_bind_group(0, &style_bg, &[(i as u32) * (STYLE_PARAMS_STRIDE as u32)]);
                        cp.dispatch_workgroups(cw.div_ceil(8), vd.div_ceil(4), ch.div_ceil(8));
                    }
                }
            }
        }

        // PASS 1: Shadow Map
        if style.shadows {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rp.set_pipeline(&self.pipe_shadow);
            if !z16_ready && !frame.terrain_idxs.is_empty() {
                rp.set_bind_group(0, &shadow_terrain_bg, &[]);
                rp.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
                rp.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..frame.terrain_idxs.len() as u32, 0, 0..1);
            }
            if !flat_meshes.is_empty() {
                rp.set_bind_group(0, &shadow_mesh_bg, &[]);
                rp.set_vertex_buffer(0, self.static_vbuf.slice(..));
                rp.set_index_buffer(self.static_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for (k, &(s, e)) in batch_ranges.iter().enumerate() {
                    if e > s {
                        let sl = self.mesh_slices[k];
                        rp.draw_indexed(sl.first_index..sl.first_index + sl.index_count, 0, s..e);
                    }
                }
            }
        }

        // PASS 2: Hard-Z Geometry (Z=16 Surface-Nets OR L4 Terrain + Parametric Heroes/Shrines + Hard-Z Surfels)
        //         (In L4 mode, also draws Tier 3 Vol-3DGS right here in the same pass!)
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("geometry_and_volumetrics_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ss_color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.ss_depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if z16_ready {
                if self.z16_icount > 0 {
                    if let (Some(bg), Some(vb), Some(ib)) =
                        (z16_mesh_bg.as_ref(), self.z16_vbuf.as_ref(), self.z16_ibuf.as_ref())
                    {
                        rp.set_pipeline(&self.pipe_field_z16);
                        rp.set_bind_group(0, bg, &[]);
                        rp.set_vertex_buffer(0, vb.slice(..));
                        rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..self.z16_icount, 0, 0..1);
                    }
                }
            } else if !frame.terrain_idxs.is_empty() {
                rp.set_pipeline(&self.pipe_mesh);
                rp.set_bind_group(0, &mesh_terrain_bg, &[]);
                rp.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
                rp.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..frame.terrain_idxs.len() as u32, 0, 0..1);
            }

            if !flat_meshes.is_empty() {
                rp.set_pipeline(&self.pipe_mesh);
                rp.set_bind_group(0, &mesh_props_bg, &[]);
                rp.set_vertex_buffer(0, self.static_vbuf.slice(..));
                rp.set_index_buffer(self.static_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for (k, &(s, e)) in batch_ranges.iter().enumerate() {
                    if e > s {
                        let sl = self.mesh_slices[k];
                        rp.draw_indexed(sl.first_index..sl.first_index + sl.index_count, 0, s..e);
                    }
                }
            }
            if !frame.surfel_refs.is_empty() {
                rp.set_pipeline(&self.pipe_surfel);
                rp.set_bind_group(0, &surfel_bg, &[]);
                rp.draw(0..6, 0..frame.surfel_refs.len() as u32);
            }
            if !z16_ready && !frame.vol_splats.is_empty() {
                rp.set_pipeline(&self.pipe_vol);
                rp.set_bind_group(0, &vol_bg, &[]);
                rp.draw(0..6, 0..frame.vol_splats.len() as u32);
            }
        }

        // PASS 2B (Z=16 only): Depth-Tested Volumetric Fluid & Blackbody Gas Raymarch Overlay
        if z16_ready {
            if let Some(ref bg) = z16_vol_bg {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("volumetric_overlay_z16_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ss_color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                rp.set_pipeline(&self.pipe_vol_z16);
                rp.set_bind_group(0, bg, &[]);
                rp.draw(0..3, 0..1);
            }
            // PASS 2C (Z=16 only): Depth-Tested Dynamic Volumetric Splats (Actuator beams, Time Domes, spells, motes)
            if !frame.vol_splats.is_empty() {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("volumetric_splats_z16_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ss_color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.ss_depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                rp.set_pipeline(&self.pipe_vol);
                rp.set_bind_group(0, &vol_bg, &[]);
                rp.draw(0..6, 0..frame.vol_splats.len() as u32);
            }
        }

        // PASS 4: Stage A 9-Step Cool-Indigo / Warm-Cream 2× SS Resolve
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resolve_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.art_color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.pipe_resolve);
            rp.set_bind_group(0, &resolve_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // PASS 5 + 6: Stage B Blit + Crisp Native-Res Action-MMO HUD & Minimap
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_and_hud_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.pipe_blit);
            rp.set_bind_group(0, &blit_bg, &[]);
            rp.draw(0..3, 0..1);

            if hud_vert_count > 0 {
                rp.set_pipeline(&self.pipe_hud);
                rp.set_bind_group(0, &hud_bg, &[]);
                rp.set_vertex_buffer(0, self.hud_vbuf.slice(..));
                rp.draw(0..hud_vert_count, 0..1);
            }
        }

        self.queue.submit(Some(enc.finish()));
        (art_w, art_h)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_rgba(
        &mut self,
        w: u32,
        h: u32,
        style: &RenderStyle,
        atmo: Atmosphere,
        camera: &MobaCamera,
        mats: &CameraMatrices,
        sim_time: f32,
        frame: &AssembledFrame,
        hud: Option<&HudBatch>,
    ) -> (Vec<u8>, f64, (u32, u32)) {
        let (tex, view) = self.create_output_texture(w, h);
        let t0 = Instant::now();
        let art_res = self.render_into(&view, w, h, style, atmo, camera, mats, sim_time, frame, hud);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rgba = self.readback_rgba(&tex, w, h);
        (rgba, gpu_ms, art_res)
    }

    pub fn readback_rgba(&self, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
        let row_bytes = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let s = (y * padded) as usize;
            out.extend_from_slice(&mapped[s..s + row_bytes as usize]);
        }
        drop(mapped);
        staging.unmap();
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_gpu(
        &mut self,
        w: u32,
        h: u32,
        style: &RenderStyle,
        atmo: Atmosphere,
        camera: &MobaCamera,
        mats: &CameraMatrices,
        frame: &AssembledFrame,
        hud: Option<&HudBatch>,
        warmup: usize,
        iters: usize,
    ) -> (f64, f64, f64) {
        let (_tex, view) = self.create_output_texture(w, h);
        for i in 0..warmup {
            self.render_into(&view, w, h, style, atmo, camera, mats, i as f32 * 0.016, frame, hud);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let mut samples = Vec::with_capacity(iters);
        for i in 0..iters {
            let t0 = Instant::now();
            self.render_into(&view, w, h, style, atmo, camera, mats, (warmup + i) as f32 * 0.016, frame, hud);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = samples[samples.len() / 2];
        let p99 = samples[((samples.len() as f64) * 0.99).floor().min((samples.len() - 1) as f64) as usize];
        let mean = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
        (p50, p99, mean)
    }
}
