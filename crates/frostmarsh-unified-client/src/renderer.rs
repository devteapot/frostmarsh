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
use crate::palette::{palette_resolve_64, palette_tex_bytes_at};
use crate::surfel_gen::{SplatPlacement, SplatProto, SplatRef, SurfelProtoBank};
use crate::world_bridge::{AssembledFrame, VolSplat};
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

#[derive(Clone, Copy, Debug)]
pub struct Atmosphere {
    pub weather_kind: u8,
    pub day_time: f32,
    pub wind_x: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            weather_kind: 0,
            day_time: 0.25,
            wind_x: 0.65,
        }
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

    shadow_bgl: wgpu::BindGroupLayout,
    mesh_bgl: wgpu::BindGroupLayout,
    surfel_bgl: wgpu::BindGroupLayout,
    vol_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    hud_bgl: wgpu::BindGroupLayout,

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
        let surfel_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surfel_instanced"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/surfel_instanced.wgsl"
            ))),
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
        let mesh_inst_cap = 16_384;
        let place_cap = 16_384;
        let ref_cap = 524_288;
        let vol_cap = 8_192;
        let hud_vcap = 32_768;

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
            shadow_bgl,
            mesh_bgl,
            surfel_bgl,
            vol_bgl,
            resolve_bgl,
            blit_bgl,
            hud_bgl,
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
        })
    }

    pub fn sun_dir(atmo: Atmosphere) -> Vec3 {
        let ang = atmo.day_time * std::f32::consts::TAU * 0.35;
        Vec3::new(0.52 * ang.cos() + 0.15, 0.72, -0.44 - 0.15 * ang.sin()).normalize()
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
        let (sun_col, sky_col, fog_col) = match atmo.weather_kind {
            WEATHER_BLIZZARD => ([0.84, 0.94, 1.08], [0.42, 0.56, 0.78], [0.22, 0.30, 0.44]),
            WEATHER_SQUALL => ([0.78, 0.84, 0.92], [0.30, 0.40, 0.56], [0.16, 0.21, 0.30]),
            WEATHER_EMBER => ([1.14, 0.84, 0.62], [0.40, 0.32, 0.46], [0.22, 0.16, 0.26]),
            _ => ([1.05, 0.96, 0.84], [0.34, 0.48, 0.68], [0.16, 0.21, 0.32]),
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
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 1.10],
            sun_col: [sun_col[0], sun_col[1], sun_col[2], if style.shadows { 0.72 } else { 0.0 }],
            sky_col: [sky_col[0], sky_col[1], sky_col[2], 120.0],
            bounce_col: [0.22, 0.25, 0.38, 280.0],
            fog_col: [fog_col[0], fog_col[1], fog_col[2], atmo.wind_x],
            point_pos: frame.point_pos,
            point_col: frame.point_col,
        };
        self.queue.write_buffer(&self.scene_ubuf, 0, bytemuck::bytes_of(&scene_u));

        let (pal_arr, pal_n) = palette_resolve_64();
        let resolve_u = ResolveUniforms {
            low_res: [art_w as f32, art_h as f32],
            ss: ss as f32,
            outline: if style.outlines { 1.0 } else { 0.0 },
            quantise: if style.crisp_resolve { 1.0 } else { 0.0 },
            dither: style.dither,
            crisp: if style.crisp_resolve { 1.0 } else { 0.0 },
            palette_count: pal_n,
            background: [fog_col[0], fog_col[1], fog_col[2], 1.0],
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

        let mut enc = self.device.create_command_encoder(&Default::default());

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
            if !frame.terrain_idxs.is_empty() {
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

        // PASS 2 + 3: Hard-Z Geometry (Terrain + Parametric Heroes/Shrines + Hard-Z Surfels)
        //           + Depth-Tested Tier 3 Hybrid Soft Volumetric 3DGS (Fire, Smoke, Steam, Wisps)
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

            rp.set_pipeline(&self.pipe_mesh);
            if !frame.terrain_idxs.is_empty() {
                rp.set_bind_group(0, &mesh_terrain_bg, &[]);
                rp.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
                rp.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..frame.terrain_idxs.len() as u32, 0, 0..1);
            }
            if !flat_meshes.is_empty() {
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
            if !frame.vol_splats.is_empty() {
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
