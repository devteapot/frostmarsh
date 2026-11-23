//! Unified `wgpu 30.0.2` Head-to-Head True 3D Renderer hosting BOTH:
//!   - **Renderer A (`Raymarch`)**: Sparse `8×8×8` Brick DDA + Trilinear Smooth Isosurface Raymarch
//!     with exact `@builtin(frag_depth)` + depth-tested 3D Parametric Hero Knight compositing.
//!   - **Renderer B (`SurfaceNets`)**: Smooth Surface-Nets 3D Isosurface Mesh (`field_mesh.wgsl`)
//!     + 3D Parametric Hero Knights -> Depth-Tested Volumetric Fluid & Gas Overlay (`volumetric_overlay.wgsl`).
//!
//! BOTH renderers consume the EXACT SAME two `RGBA8Unorm` 3D GPU textures (`density_mat`,
//! `albedo_thermal`) and coarse `8×8×8` brick occupancy texture, with zero hardcoded species rules!

use crate::camera::Camera3D;
use crate::hero_mesh::{HeroInstance, HeroMesh, HeroVertex};
use crate::surface_nets_3d::{SurfaceNetsWorld, SurfaceVertex};
use crate::volume::{BrickOccupancy3D, WorldVolume3D, BRICK_SIZE};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::borrow::Cow;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode {
    Raymarch,
    SurfaceNets,
}

impl RenderMode {
    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Raymarch => "Renderer A: Sparse Brick Raymarch",
            RenderMode::SurfaceNets => "Renderer B: Smooth Surface-Nets + Vol Overlay",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            RenderMode::Raymarch => RenderMode::SurfaceNets,
            RenderMode::SurfaceNets => RenderMode::Raymarch,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sun_col: [f32; 4],
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    vol_dims: [f32; 4],
    brick_dims: [f32; 4],
    viewport: [f32; 4],
}

pub struct BenchResult {
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub fps_p50: f64,
}

pub struct HeadToHeadRenderer {
    pub adapter_name: String,
    pub adapter_backend: String,
    pub out_format: wgpu::TextureFormat,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    vol_dims: [f32; 3],
    brick_dims: [f32; 3],

    // Uniform buffers (separate for tonemapped Renderer A vs linear HDR Renderer B so both can run cleanly)
    scene_ubuf_a: wgpu::Buffer,
    scene_ubuf_b: wgpu::Buffer,

    // Shared 3D textures + sampler
    _density_tex: wgpu::Texture,
    _albedo_tex: wgpu::Texture,
    _brick_tex: wgpu::Texture,
    density_view: wgpu::TextureView,
    albedo_view: wgpu::TextureView,
    brick_view: wgpu::TextureView,
    vol_samp: wgpu::Sampler,

    // Pipelines & layouts
    pipe_raymarch: wgpu::RenderPipeline,
    pipe_field_mesh: wgpu::RenderPipeline,
    pipe_vol_overlay: wgpu::RenderPipeline,
    pipe_hero_a: wgpu::RenderPipeline,
    pipe_hero_b: wgpu::RenderPipeline,

    bg_vol_a: wgpu::BindGroup,
    bg_vol_b: wgpu::BindGroup,
    overlay_bgl: wgpu::BindGroupLayout,

    // Surface-Nets GPU geometry buffers
    sn_vbuf: wgpu::Buffer,
    sn_ibuf: wgpu::Buffer,
    pub sn_index_count: u32,
    pub sn_vertex_count: u32,

    // Hero Knight GPU geometry & instance buffers
    hero_vbuf: wgpu::Buffer,
    hero_ibuf: wgpu::Buffer,
    hero_index_count: u32,
    hero_inst_buf: wgpu::Buffer,
    hero_inst_count: u32,

    // Resolution-dependent targets (depth + linear HDR buffer for Renderer B)
    target_w: u32,
    target_h: u32,
    depth_view: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    bg_overlay: wgpu::BindGroup,
}

fn create_3d_rgba8_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    w: usize,
    d: usize,
    h: usize,
    data: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: w as u32,
        height: d as u32,
        depth_or_array_layers: h as u32,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((w * 4) as u32),
            rows_per_image: Some(d as u32),
        },
        size,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    (tex, view)
}

fn make_targets(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: w.max(1),
        height: h.max(1),
        depth_or_array_layers: 1,
    };
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene_depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene_hdr"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    (
        depth_tex.create_view(&Default::default()),
        hdr_tex.create_view(&Default::default()),
    )
}

impl HeadToHeadRenderer {
    pub async fn new_headless(
        volume: &WorldVolume3D,
        bricks: &BrickOccupancy3D,
        sn_world: &SurfaceNetsWorld,
        hero_mesh: &HeroMesh,
        hero_instances: &[HeroInstance],
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
        Self::from_adapter(
            adapter,
            wgpu::TextureFormat::Rgba8Unorm,
            volume,
            bricks,
            sn_world,
            hero_mesh,
            hero_instances,
        )
        .await
    }

    pub async fn from_adapter(
        adapter: wgpu::Adapter,
        out_format: wgpu::TextureFormat,
        volume: &WorldVolume3D,
        bricks: &BrickOccupancy3D,
        sn_world: &SurfaceNetsWorld,
        hero_mesh: &HeroMesh,
        hero_instances: &[HeroInstance],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let adapter_backend = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("frostmarsh_3d_sandbox_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await?;

        // Upload the two 256x16x256 3D textures + 32x2x32 8x8x8 coarse occupancy brick texture
        let (density_tex, density_view) = create_3d_rgba8_texture(
            &device,
            &queue,
            "density_mat_3d",
            volume.width,
            volume.depth,
            volume.height,
            &volume.density_mat,
        );
        let (albedo_tex, albedo_view) = create_3d_rgba8_texture(
            &device,
            &queue,
            "albedo_thermal_3d",
            volume.width,
            volume.depth,
            volume.height,
            &volume.albedo_thermal,
        );
        let (brick_tex, brick_view) = create_3d_rgba8_texture(
            &device,
            &queue,
            "brick_occupancy_3d",
            bricks.bx_count,
            bricks.by_count,
            bricks.bz_count,
            &bricks.data,
        );

        let vol_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vol_trilinear_samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let scene_ubuf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_ubuf_a"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_ubuf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_ubuf_b"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared volume bind group layout (used by Raymarch, FieldMesh, and HeroMesh)
        let vol_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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

        let make_vol_bg = |label: &str, ubuf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &vol_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&density_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&brick_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&vol_samp) },
                ],
            })
        };
        let bg_vol_a = make_vol_bg("bg_vol_a", &scene_ubuf_a);
        let bg_vol_b = make_vol_bg("bg_vol_b", &scene_ubuf_b);

        // Compile WGSL shader modules
        let raymarch_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_raymarch"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/volumetric_raymarch.wgsl"))),
        });
        let field_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_mesh"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/field_mesh.wgsl"))),
        });
        let overlay_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_overlay"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/volumetric_overlay.wgsl"))),
        });
        let hero_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hero_mesh"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/hero_mesh.wgsl"))),
        });

        let vol_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol_pl"),
            bind_group_layouts: &[Some(&vol_bgl)],
            immediate_size: 0,
        });
        let overlay_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_pl"),
            bind_group_layouts: &[Some(&overlay_bgl)],
            immediate_size: 0,
        });

        // Pipeline 1: Renderer A (Raymarch fullscreen writing color + exact @builtin(frag_depth))
        let pipe_raymarch = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_raymarch"),
            layout: Some(&vol_pl),
            vertex: wgpu::VertexState {
                module: &raymarch_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &raymarch_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: out_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Pipeline 2: Renderer B Pass 1 (Surface-Nets 3D solid mesh -> linear HDR + hardware depth)
        let sn_vbl = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SurfaceVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        })];
        let pipe_field_mesh = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_field_mesh"),
            layout: Some(&vol_pl),
            vertex: wgpu::VertexState {
                module: &field_mesh_mod,
                entry_point: Some("vs_main"),
                buffers: &sn_vbl,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &field_mesh_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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

        // Pipeline 3: Renderer B Pass 2 (Depth-tested Volumetric Fluid/Gas Overlay + Sky + Tonemap)
        let pipe_vol_overlay = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_vol_overlay"),
            layout: Some(&overlay_pl),
            vertex: wgpu::VertexState {
                module: &overlay_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_mod,
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

        // Pipelines 4 & 5: Shared 3D Hero Knight (Renderer A outputs to out_format, Renderer B outputs to Rgba16Float)
        let hero_vbl = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<HeroVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3,
                    1 => Float32x3,
                    2 => Float32x4,
                    3 => Float32x4
                ],
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<HeroInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![
                    4 => Float32x4,
                    5 => Float32x4
                ],
            }),
        ];
        let make_hero_pipe = |label: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&vol_pl),
                vertex: wgpu::VertexState {
                    module: &hero_mod,
                    entry_point: Some("vs_main"),
                    buffers: &hero_vbl,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &hero_mod,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
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
            })
        };
        let pipe_hero_a = make_hero_pipe("pipe_hero_a", out_format);
        let pipe_hero_b = make_hero_pipe("pipe_hero_b", wgpu::TextureFormat::Rgba16Float);

        // Upload Surface-Nets geometry buffers
        let (sn_verts, sn_idxs) = sn_world.flatten_buffers();
        let sn_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn_vbuf"),
            size: (sn_verts.len().max(1) * std::mem::size_of::<SurfaceVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sn_ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn_ibuf"),
            size: (sn_idxs.len().max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !sn_verts.is_empty() {
            queue.write_buffer(&sn_vbuf, 0, bytemuck::cast_slice(&sn_verts));
        }
        if !sn_idxs.is_empty() {
            queue.write_buffer(&sn_ibuf, 0, bytemuck::cast_slice(&sn_idxs));
        }

        // Upload Hero Knight geometry & instance buffers
        let hero_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hero_vbuf"),
            size: (hero_mesh.vertices.len() * std::mem::size_of::<HeroVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hero_ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hero_ibuf"),
            size: (hero_mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hero_inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hero_inst_buf"),
            size: (hero_instances.len().max(1) * std::mem::size_of::<HeroInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&hero_vbuf, 0, bytemuck::cast_slice(&hero_mesh.vertices));
        queue.write_buffer(&hero_ibuf, 0, bytemuck::cast_slice(&hero_mesh.indices));
        if !hero_instances.is_empty() {
            queue.write_buffer(&hero_inst_buf, 0, bytemuck::cast_slice(hero_instances));
        }

        let (depth_view, hdr_view) = make_targets(&device, 1280, 720);
        let bg_overlay = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_overlay"),
            layout: &overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scene_ubuf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&brick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&vol_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&depth_view) },
            ],
        });

        Ok(Self {
            adapter_name,
            adapter_backend,
            out_format,
            device,
            queue,
            vol_dims: [volume.width as f32, volume.depth as f32, volume.height as f32],
            brick_dims: [bricks.bx_count as f32, bricks.by_count as f32, bricks.bz_count as f32],
            scene_ubuf_a,
            scene_ubuf_b,
            _density_tex: density_tex,
            _albedo_tex: albedo_tex,
            _brick_tex: brick_tex,
            density_view,
            albedo_view,
            brick_view,
            vol_samp,
            pipe_raymarch,
            pipe_field_mesh,
            pipe_vol_overlay,
            pipe_hero_a,
            pipe_hero_b,
            bg_vol_a,
            bg_vol_b,
            overlay_bgl,
            sn_vbuf,
            sn_ibuf,
            sn_index_count: sn_idxs.len() as u32,
            sn_vertex_count: sn_verts.len() as u32,
            hero_vbuf,
            hero_ibuf,
            hero_index_count: hero_mesh.indices.len() as u32,
            hero_inst_buf,
            hero_inst_count: hero_instances.len() as u32,
            target_w: 1280,
            target_h: 720,
            depth_view,
            hdr_view,
            bg_overlay,
        })
    }

    fn ensure_targets(&mut self, w: u32, h: u32) {
        if self.target_w == w && self.target_h == h {
            return;
        }
        let (depth_view, hdr_view) = make_targets(&self.device, w, h);
        self.bg_overlay = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_overlay"),
            layout: &self.overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.albedo_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.brick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.vol_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&depth_view) },
            ],
        });
        self.depth_view = depth_view;
        self.hdr_view = hdr_view;
        self.target_w = w;
        self.target_h = h;
    }

    fn build_uniforms(&self, cam: &Camera3D, w: u32, h: u32, sim_time: f32, tonemap: f32) -> SceneUniforms {
        let aspect = w.max(1) as f32 / h.max(1) as f32;
        let mats = cam.matrices(aspect);
        // Golden alpine afternoon sun shining from South-East to carve deep shadows under bridge arches,
        // illuminate the plunging waterfall, and frame the dramatic dark basalt cave entrance!
        let sun_dir = Vec3::new(0.56, 0.54, 0.63).normalize();
        SceneUniforms {
            view_proj: mats.view_proj.to_cols_array_2d(),
            inv_view_proj: mats.inv_view_proj.to_cols_array_2d(),
            cam_pos: [cam.pos.x, cam.pos.y, cam.pos.z, sim_time],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 1.08],
            sun_col: [1.85, 1.62, 1.24, 1.0],
            sky_top: [0.06, 0.20, 0.54, 1.0],
            sky_horizon: [0.36, 0.54, 0.78, 1.0],
            vol_dims: [self.vol_dims[0], self.vol_dims[1], self.vol_dims[2], tonemap],
            brick_dims: [
                self.brick_dims[0],
                self.brick_dims[1],
                self.brick_dims[2],
                BRICK_SIZE as f32,
            ],
            viewport: [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
        }
    }

    pub fn render_into(
        &mut self,
        out_view: &wgpu::TextureView,
        w: u32,
        h: u32,
        mode: RenderMode,
        cam: &Camera3D,
        sim_time: f32,
    ) {
        self.ensure_targets(w, h);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headtohead_enc"),
        });

        match mode {
            RenderMode::Raymarch => {
                let uni = self.build_uniforms(cam, w, h, sim_time, 1.0);
                self.queue.write_buffer(&self.scene_ubuf_a, 0, bytemuck::bytes_of(&uni));

                // Pass 1: Sparse 8x8x8 Brick DDA Raymarch (writes color + exact @builtin(frag_depth))
                // followed immediately in the SAME render pass by depth-tested 3D Hero Knights!
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("raymarch_and_heroes"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: out_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                rp.set_pipeline(&self.pipe_raymarch);
                rp.set_bind_group(0, &self.bg_vol_a, &[]);
                rp.draw(0..3, 0..1);

                if self.hero_inst_count > 0 {
                    rp.set_pipeline(&self.pipe_hero_a);
                    rp.set_bind_group(0, &self.bg_vol_a, &[]);
                    rp.set_vertex_buffer(0, self.hero_vbuf.slice(..));
                    rp.set_vertex_buffer(1, self.hero_inst_buf.slice(..));
                    rp.set_index_buffer(self.hero_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..self.hero_index_count, 0, 0..self.hero_inst_count);
                }
            }
            RenderMode::SurfaceNets => {
                let uni = self.build_uniforms(cam, w, h, sim_time, 0.0);
                self.queue.write_buffer(&self.scene_ubuf_b, 0, bytemuck::bytes_of(&uni));

                // Pass 1: Rasterize Smooth Surface-Nets solid mesh + 3D Hero Knights into HDR + Depth
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("surface_nets_and_heroes"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.hdr_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    if self.sn_index_count > 0 {
                        rp.set_pipeline(&self.pipe_field_mesh);
                        rp.set_bind_group(0, &self.bg_vol_b, &[]);
                        rp.set_vertex_buffer(0, self.sn_vbuf.slice(..));
                        rp.set_index_buffer(self.sn_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..self.sn_index_count, 0, 0..1);
                    }

                    if self.hero_inst_count > 0 {
                        rp.set_pipeline(&self.pipe_hero_b);
                        rp.set_bind_group(0, &self.bg_vol_b, &[]);
                        rp.set_vertex_buffer(0, self.hero_vbuf.slice(..));
                        rp.set_vertex_buffer(1, self.hero_inst_buf.slice(..));
                        rp.set_index_buffer(self.hero_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..self.hero_index_count, 0, 0..self.hero_inst_count);
                    }
                }

                // Pass 2: Depth-Tested Volumetric Fluid & Gas Overlay + Sky + ACES Tonemap
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("volumetric_overlay"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: out_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    rp.set_pipeline(&self.pipe_vol_overlay);
                    rp.set_bind_group(0, &self.bg_overlay, &[]);
                    rp.draw(0..3, 0..1);
                }
            }
        }

        self.queue.submit(Some(enc.finish()));
    }

    fn create_output_texture(&self, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless_out"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
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

    pub fn render_to_rgba(
        &mut self,
        w: u32,
        h: u32,
        mode: RenderMode,
        cam: &Camera3D,
        sim_time: f32,
    ) -> Vec<u8> {
        let (tex, view) = self.create_output_texture(w, h);
        self.render_into(&view, w, h, mode, cam, sim_time);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let row_bytes = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
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
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
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

    pub fn benchmark(
        &mut self,
        w: u32,
        h: u32,
        mode: RenderMode,
        cam: &Camera3D,
        warmup: usize,
        iters: usize,
    ) -> BenchResult {
        let (_tex, view) = self.create_output_texture(w, h);
        for i in 0..warmup {
            self.render_into(&view, w, h, mode, cam, i as f32 * 0.016);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let mut samples = Vec::with_capacity(iters);
        for i in 0..iters {
            let t0 = Instant::now();
            self.render_into(&view, w, h, mode, cam, (warmup + i) as f32 * 0.016);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = samples[samples.len() / 2];
        let p99 = samples[((samples.len() as f64) * 0.99)
            .floor()
            .min((samples.len() - 1) as f64) as usize];
        let mean = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
        BenchResult {
            p50_ms: p50,
            p99_ms: p99,
            mean_ms: mean,
            fps_p50: 1000.0 / p50.max(1e-4),
        }
    }
}
