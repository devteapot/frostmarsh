#![cfg(target_arch = "wasm32")]
//! WebGPU Smooth Surface-Nets + Direct 3D-Texture Shading + Volumetric Fluid/Gas Overlay
//! + 3D Parametric Hero Knights + Stylized `Pixel-2×SS` ↔ `Clean-3D` Resolve Renderer.

use crate::camera::Camera3D;
use crate::hero_mesh::{HeroInstance, HeroMesh, HeroVertex};
use crate::surface_nets_3d::{SurfaceNetsWorld, SurfaceVertex};
use crate::volume::{BrickOccupancy3D, WorldVolume3D, BRICK_SIZE};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::borrow::Cow;

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniforms {
    viewport: [f32; 4],
    params: [f32; 4], // x = pixel_mode (0=Clean-3D, 1=Pixel-2xSS), y = pixel_scale (2.0)
}

#[cfg(target_arch = "wasm32")]
pub struct WebRenderer {
    pub adapter_name: String,
    pub adapter_backend: String,
    pub surface: wgpu::Surface<'static>,
    pub surface_cfg: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    vol_dims: [f32; 3],
    brick_dims: [f32; 3],

    scene_ubuf: wgpu::Buffer,
    post_ubuf: wgpu::Buffer,

    _density_tex: wgpu::Texture,
    _albedo_tex: wgpu::Texture,
    _brick_tex: wgpu::Texture,
    density_view: wgpu::TextureView,
    albedo_view: wgpu::TextureView,
    brick_view: wgpu::TextureView,
    vol_samp: wgpu::Sampler,

    vol_bgl: wgpu::BindGroupLayout,
    overlay_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,

    bg_vol: wgpu::BindGroup,
    bg_overlay: wgpu::BindGroup,
    bg_post: wgpu::BindGroup,

    pipe_field_mesh: wgpu::RenderPipeline,
    pipe_hero: wgpu::RenderPipeline,
    pipe_vol_overlay: wgpu::RenderPipeline,
    pipe_post_resolve: wgpu::RenderPipeline,

    sn_vbuf: wgpu::Buffer,
    sn_ibuf: wgpu::Buffer,
    pub sn_index_count: u32,
    pub sn_vertex_count: u32,

    hero_vbuf: wgpu::Buffer,
    hero_ibuf: wgpu::Buffer,
    hero_index_count: u32,
    hero_inst_buf: wgpu::Buffer,
    hero_inst_count: u32,

    target_w: u32,
    target_h: u32,
    depth_view: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    tonemapped_view: wgpu::TextureView,
}

#[cfg(target_arch = "wasm32")]
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

#[cfg(target_arch = "wasm32")]
fn make_targets(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
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
    let tonemapped_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene_tonemapped"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    (
        depth_tex.create_view(&Default::default()),
        hdr_tex.create_view(&Default::default()),
        tonemapped_tex.create_view(&Default::default()),
    )
}

#[cfg(target_arch = "wasm32")]
impl WebRenderer {
    pub async fn from_canvas(
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        volume: &WorldVolume3D,
        bricks: &BrickOccupancy3D,
        sn_world: &SurfaceNetsWorld,
        hero_mesh: &HeroMesh,
        hero_instances: &[HeroInstance],
    ) -> Result<Self, String> {
        let w = width.clamp(1, 4096);
        let h = height.clamp(1, 4096);

        // Try native WebGPU first; fall back to WebGL2 only if WebGPU is absent.
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(desc);

        let surface_res = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()));
        let (surface, adapter) = match surface_res {
            Ok(surf) => match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surf),
                    force_fallback_adapter: false,
                    ..Default::default()
                })
                .await
            {
                Ok(adp) => (surf, adp),
                Err(webgpu_err) => {
                    let mut gl_desc = wgpu::InstanceDescriptor::new_without_display_handle();
                    gl_desc.backends = wgpu::Backends::GL;
                    let gl_instance = wgpu::Instance::new(gl_desc);
                    let gl_surf = gl_instance
                        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                        .map_err(|e| {
                            format!(
                                "WebGPU unavailable ({webgpu_err}); WebGL fallback surface failed: {e}"
                            )
                        })?;
                    let gl_adp = gl_instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            compatible_surface: Some(&gl_surf),
                            force_fallback_adapter: false,
                            ..Default::default()
                        })
                        .await
                        .map_err(|e| {
                            format!("No WebGPU ({webgpu_err}) or WebGL2 ({e}) GPU adapter available")
                        })?;
                    (gl_surf, gl_adp)
                }
            },
            Err(e) => return Err(format!("Failed to create WebGPU surface: {e}")),
        };

        let info = adapter.get_info();
        let adapter_name = if info.name.trim().is_empty() {
            format!("{:?} GPU", info.device_type)
        } else {
            info.name.clone()
        };
        let adapter_backend = format!("{:?}", info.backend);

        let limits = if info.backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("frostmarsh_webgpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to request GPU device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let out_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let mut surface_cfg = surface
            .get_default_config(&adapter, w, h)
            .ok_or_else(|| "Surface incompatible with adapter".to_string())?;
        surface_cfg.format = out_format;
        surface_cfg.alpha_mode = caps.alpha_modes[0];
        surface.configure(&device, &surface_cfg);

        // Upload 3D textures (density_mat, albedo_thermal, brick_occupancy)
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

        let scene_ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_ubuf"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let post_ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_ubuf"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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

        let field_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_mesh"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/field_mesh.wgsl"
            ))),
        });
        let hero_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hero_mesh"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/hero_mesh.wgsl"
            ))),
        });
        let overlay_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_overlay"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/volumetric_overlay.wgsl"
            ))),
        });
        let post_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_resolve"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/post_resolve.wgsl"
            ))),
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
        let post_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_pl"),
            bind_group_layouts: &[Some(&post_bgl)],
            immediate_size: 0,
        });

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
        let pipe_hero = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_hero"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
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
        });

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

        let pipe_post_resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_post_resolve"),
            layout: Some(&post_pl),
            vertex: wgpu::VertexState {
                module: &post_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &post_mod,
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

        // Upload Surface-Nets geometry
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

        // Upload Hero Knight geometry & instances
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

        let (depth_view, hdr_view, tonemapped_view) = make_targets(&device, w, h);

        let bg_vol = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_vol"),
            layout: &vol_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&brick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&vol_samp) },
            ],
        });
        let bg_overlay = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_overlay"),
            layout: &overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&brick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&vol_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&depth_view) },
            ],
        });
        let bg_post = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_post"),
            layout: &post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: post_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tonemapped_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&depth_view) },
            ],
        });

        Ok(Self {
            adapter_name,
            adapter_backend,
            surface,
            surface_cfg,
            device,
            queue,
            vol_dims: [volume.width as f32, volume.depth as f32, volume.height as f32],
            brick_dims: [
                bricks.bx_count as f32,
                bricks.by_count as f32,
                bricks.bz_count as f32,
            ],
            scene_ubuf,
            post_ubuf,
            _density_tex: density_tex,
            _albedo_tex: albedo_tex,
            _brick_tex: brick_tex,
            density_view,
            albedo_view,
            brick_view,
            vol_samp,
            vol_bgl,
            overlay_bgl,
            post_bgl,
            bg_vol,
            bg_overlay,
            bg_post,
            pipe_field_mesh,
            pipe_hero,
            pipe_vol_overlay,
            pipe_post_resolve,
            sn_vbuf,
            sn_ibuf,
            sn_index_count: sn_idxs.len() as u32,
            sn_vertex_count: sn_verts.len() as u32,
            hero_vbuf,
            hero_ibuf,
            hero_index_count: hero_mesh.indices.len() as u32,
            hero_inst_buf,
            hero_inst_count: hero_instances.len() as u32,
            target_w: w,
            target_h: h,
            depth_view,
            hdr_view,
            tonemapped_view,
        })
    }

    /// Hot-swap the GPU 3D volume textures and Surface-Nets mesh live in browser!
    pub fn upload_volume(
        &mut self,
        volume: &WorldVolume3D,
        bricks: &BrickOccupancy3D,
        sn_world: &SurfaceNetsWorld,
    ) {
        let (density_tex, density_view) = create_3d_rgba8_texture(
            &self.device,
            &self.queue,
            "density_mat_3d",
            volume.width,
            volume.depth,
            volume.height,
            &volume.density_mat,
        );
        let (albedo_tex, albedo_view) = create_3d_rgba8_texture(
            &self.device,
            &self.queue,
            "albedo_thermal_3d",
            volume.width,
            volume.depth,
            volume.height,
            &volume.albedo_thermal,
        );
        let (brick_tex, brick_view) = create_3d_rgba8_texture(
            &self.device,
            &self.queue,
            "brick_occupancy_3d",
            bricks.bx_count,
            bricks.by_count,
            bricks.bz_count,
            &bricks.data,
        );

        let (sn_verts, sn_idxs) = sn_world.flatten_buffers();
        let sn_vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn_vbuf"),
            size: (sn_verts.len().max(1) * std::mem::size_of::<SurfaceVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sn_ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn_ibuf"),
            size: (sn_idxs.len().max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !sn_verts.is_empty() {
            self.queue.write_buffer(&sn_vbuf, 0, bytemuck::cast_slice(&sn_verts));
        }
        if !sn_idxs.is_empty() {
            self.queue.write_buffer(&sn_ibuf, 0, bytemuck::cast_slice(&sn_idxs));
        }

        self.vol_dims = [volume.width as f32, volume.depth as f32, volume.height as f32];
        self.brick_dims = [
            bricks.bx_count as f32,
            bricks.by_count as f32,
            bricks.bz_count as f32,
        ];
        self._density_tex = density_tex;
        self._albedo_tex = albedo_tex;
        self._brick_tex = brick_tex;
        self.density_view = density_view;
        self.albedo_view = albedo_view;
        self.brick_view = brick_view;
        self.sn_vbuf = sn_vbuf;
        self.sn_ibuf = sn_ibuf;
        self.sn_vertex_count = sn_verts.len() as u32;
        self.sn_index_count = sn_idxs.len() as u32;
        self.rebuild_bind_groups();
    }

    fn rebuild_bind_groups(&mut self) {
        self.bg_vol = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_vol"),
            layout: &self.vol_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.albedo_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.brick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.vol_samp) },
            ],
        });
        self.bg_overlay = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_overlay"),
            layout: &self.overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.albedo_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.brick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.vol_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
            ],
        });
        self.bg_post = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_post"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.post_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.tonemapped_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
            ],
        });
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.clamp(1, 4096);
        let h = height.clamp(1, 4096);
        if self.target_w == w && self.target_h == h {
            return;
        }
        self.surface_cfg.width = w;
        self.surface_cfg.height = h;
        self.surface.configure(&self.device, &self.surface_cfg);

        let (depth_view, hdr_view, tonemapped_view) = make_targets(&self.device, w, h);
        self.depth_view = depth_view;
        self.hdr_view = hdr_view;
        self.tonemapped_view = tonemapped_view;
        self.target_w = w;
        self.target_h = h;
        self.rebuild_bind_groups();
    }

    pub fn render_frame(&mut self, cam: &Camera3D, sim_time: f32, pixel_art: bool) {
        let surf_tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_cfg);
                return;
            }
            _ => return,
        };
        let out_view = surf_tex.texture.create_view(&Default::default());

        let w = self.target_w;
        let h = self.target_h;
        let aspect = w.max(1) as f32 / h.max(1) as f32;
        let mats = cam.matrices(aspect);
        let sun_dir = Vec3::new(0.56, 0.54, 0.63).normalize();

        let scene_uni = SceneUniforms {
            view_proj: mats.view_proj.to_cols_array_2d(),
            inv_view_proj: mats.inv_view_proj.to_cols_array_2d(),
            cam_pos: [cam.pos.x, cam.pos.y, cam.pos.z, sim_time],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 1.08],
            sun_col: [1.85, 1.62, 1.24, 1.0],
            sky_top: [0.06, 0.20, 0.54, 1.0],
            sky_horizon: [0.36, 0.54, 0.78, 1.0],
            vol_dims: [self.vol_dims[0], self.vol_dims[1], self.vol_dims[2], 0.0],
            brick_dims: [
                self.brick_dims[0],
                self.brick_dims[1],
                self.brick_dims[2],
                BRICK_SIZE as f32,
            ],
            viewport: [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
        };
        self.queue
            .write_buffer(&self.scene_ubuf, 0, bytemuck::bytes_of(&scene_uni));

        let post_uni = PostUniforms {
            viewport: [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
            params: [if pixel_art { 1.0 } else { 0.0 }, 2.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.post_ubuf, 0, bytemuck::bytes_of(&post_uni));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frostmarsh_webgpu_enc"),
            });

        // Pass 1: Smooth Surface-Nets solid mesh + 3D Hero Knights -> HDR + Depth
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
                rp.set_bind_group(0, &self.bg_vol, &[]);
                rp.set_vertex_buffer(0, self.sn_vbuf.slice(..));
                rp.set_index_buffer(self.sn_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.sn_index_count, 0, 0..1);
            }

            if self.hero_inst_count > 0 {
                rp.set_pipeline(&self.pipe_hero);
                rp.set_bind_group(0, &self.bg_vol, &[]);
                rp.set_vertex_buffer(0, self.hero_vbuf.slice(..));
                rp.set_vertex_buffer(1, self.hero_inst_buf.slice(..));
                rp.set_index_buffer(self.hero_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.hero_index_count, 0, 0..self.hero_inst_count);
            }
        }

        // Pass 2: Depth-Tested Volumetric Fluid/Gas Overlay + Sky + ACES Tonemap
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.tonemapped_view,
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

        // Pass 3: Pixel-2xSS <-> Clean-3D Resolve -> Browser Swapchain
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
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
            rp.set_pipeline(&self.pipe_post_resolve);
            rp.set_bind_group(0, &self.bg_post, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(Some(enc.finish()));
        self.queue.present(surf_tex);
    }
}
