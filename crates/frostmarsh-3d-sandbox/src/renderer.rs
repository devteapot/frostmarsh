//! Unified `wgpu 30.0.2` True 3D `Z=16` Renderer hosting:
//!   - **Client-Side WGSL `NeuralMaterialHead(Z_world)` Compute Decoder** (`neural_material_decode.wgsl`)
//!     live-decoding compact `FMLT` latent bricks directly into the two `RGBA8Unorm` 3D GPU textures
//!     for 3 swappable visual styles (`Y`-key: `NORDIC PIXEL-2XSS` / `VOLCANIC EMBER` / `ARCANE THERMAL`)!
//!   - **Renderer B (`SurfaceNets`, primary stylized engine)**:
//!       1. Smooth Surface-Nets 3D Isosurface Mesh (`field_mesh.wgsl` with triplanar painterly micro-detail)
//!       2. Hard-Z GPU-Instanced Sculpted Pine Canopy & Snow-Pine Surfels (`pine_surfels.wgsl`)
//!       3. Depth-Tested 3D Parametric Hero Knights (`hero_mesh.wgsl`)
//!       4. Depth-Tested Volumetric Fluid & Gas Overlay (`volumetric_overlay.wgsl`)
//!       5. Stage A 2×SS HD Pixel-Art Resolve (`pixel_resolve.wgsl`, live `T`-key `PIXEL-2XSS` ↔ `CLEAN-3D`!)
//!   - **Renderer A (`Raymarch`, head-to-head reference)**:
//!       Sparse `8×8×8` Brick DDA + Trilinear Smooth Isosurface Raymarch + 3D Hero Knights (`G`-key toggle).

use crate::camera::Camera3D;
use crate::foliage_surfels::{PineSurfelScene, SplatPlacement, SplatProto, SplatRef};
use crate::hero_mesh::{HeroInstance, HeroMesh, HeroVertex};
use crate::neural_decoder::{
    ClientMaterialWeights, GpuDecodeBenchMetrics, GpuNeuralDecoder, LatentBrickPacket, VisualStyle,
};
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
            RenderMode::SurfaceNets => "Renderer B: Surface-Nets + Pines + Vol",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            RenderMode::Raymarch => RenderMode::SurfaceNets,
            RenderMode::SurfaceNets => RenderMode::Raymarch,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolveStyle {
    Pixel2xSS,
    Clean3D,
}

impl ResolveStyle {
    pub fn label(self) -> &'static str {
        match self {
            ResolveStyle::Pixel2xSS => "PIXEL-2XSS",
            ResolveStyle::Clean3D => "CLEAN-3D",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            ResolveStyle::Pixel2xSS => ResolveStyle::Clean3D,
            ResolveStyle::Clean3D => ResolveStyle::Pixel2xSS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
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
struct ResolveUniforms {
    out_res: [f32; 2],
    cell_scale: f32,
    outline: f32,
    quantise: f32,
    dither: f32,
    crisp: f32,
    palette_count: f32,
    depth_params: [f32; 4],
    ink_tint: [f32; 4],
    palette: [[f32; 4]; 64],
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

    // Active stylization & client neural material state
    pub visual_style: VisualStyle,
    pub resolve_style: ResolveStyle,
    neural_decoder: Option<GpuNeuralDecoder>,

    // Uniform buffers
    scene_ubuf_a: wgpu::Buffer,
    scene_ubuf_b: wgpu::Buffer,
    resolve_ubuf: wgpu::Buffer,

    // Shared 3D textures + sampler (Rgba8Unorm with STORAGE_BINDING for WGSL compute decode!)
    _density_tex: wgpu::Texture,
    albedo_tex: wgpu::Texture,
    _brick_tex: wgpu::Texture,
    density_view: wgpu::TextureView,
    albedo_view: wgpu::TextureView,
    brick_view: wgpu::TextureView,
    vol_samp: wgpu::Sampler,

    // Pipelines & layouts
    pipe_raymarch: wgpu::RenderPipeline,
    pipe_field_mesh: wgpu::RenderPipeline,
    pipe_pine_surfels: wgpu::RenderPipeline,
    pipe_vol_overlay: wgpu::RenderPipeline,
    pipe_pixel_resolve: wgpu::RenderPipeline,
    pipe_hero_a: wgpu::RenderPipeline,
    pipe_hero_b: wgpu::RenderPipeline,

    bg_vol_a: wgpu::BindGroup,
    bg_vol_b: wgpu::BindGroup,
    bg_surfels: wgpu::BindGroup,
    overlay_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,

    // Surface-Nets GPU geometry buffers
    sn_vbuf: wgpu::Buffer,
    sn_ibuf: wgpu::Buffer,
    pub sn_index_count: u32,
    pub sn_vertex_count: u32,

    // Hard-Z Pine Surfel GPU buffers
    _surfel_proto_buf: wgpu::Buffer,
    _surfel_place_buf: wgpu::Buffer,
    _surfel_ref_buf: wgpu::Buffer,
    pub surfel_draw_count: u32,
    pub pine_count: usize,
    pub snow_pine_count: usize,

    // Hero Knight GPU geometry & instance buffers
    hero_vbuf: wgpu::Buffer,
    hero_ibuf: wgpu::Buffer,
    hero_index_count: u32,
    hero_inst_buf: wgpu::Buffer,
    hero_inst_count: u32,

    // Resolution-dependent targets (depth, linear HDR, and LDR resolve source for Stage A 2xSS!)
    target_w: u32,
    target_h: u32,
    depth_view: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    resolve_src_view: wgpu::TextureView,
    bg_overlay: wgpu::BindGroup,
    bg_resolve: wgpu::BindGroup,
}

fn create_3d_rgba8_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    w: usize,
    d: usize,
    h: usize,
    data: &[u8],
    allow_storage: bool,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: w as u32,
        height: d as u32,
        depth_or_array_layers: h as u32,
    };
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if allow_storage {
        usage |= wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
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
    let resolve_src_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resolve_src_ldr"),
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
        resolve_src_tex.create_view(&Default::default()),
    )
}

impl HeadToHeadRenderer {
    pub async fn new_headless(
        volume: &WorldVolume3D,
        bricks: &BrickOccupancy3D,
        sn_world: &SurfaceNetsWorld,
        hero_mesh: &HeroMesh,
        hero_instances: &[HeroInstance],
        pines: &PineSurfelScene,
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
            pines,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn from_adapter(
        adapter: wgpu::Adapter,
        out_format: wgpu::TextureFormat,
        volume: &WorldVolume3D,
        bricks: &BrickOccupancy3D,
        sn_world: &SurfaceNetsWorld,
        hero_mesh: &HeroMesh,
        hero_instances: &[HeroInstance],
        pines: &PineSurfelScene,
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

        let (density_tex, density_view) = create_3d_rgba8_texture(
            &device,
            &queue,
            "density_mat_3d",
            volume.width,
            volume.depth,
            volume.height,
            &volume.density_mat,
            true,
        );
        let (albedo_tex, albedo_view) = create_3d_rgba8_texture(
            &device,
            &queue,
            "albedo_thermal_3d",
            volume.width,
            volume.depth,
            volume.height,
            &volume.albedo_thermal,
            true,
        );
        let (brick_tex, brick_view) = create_3d_rgba8_texture(
            &device,
            &queue,
            "brick_occupancy_3d",
            bricks.bx_count,
            bricks.by_count,
            bricks.bz_count,
            &bricks.data,
            false,
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
        let resolve_ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolve_ubuf"),
            size: std::mem::size_of::<ResolveUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared volume bind group layout (vertex+fragment so pine_surfels can sample density_tex in VS!)
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let surfel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfel_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve_bgl"),
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

        // Upload Hard-Z Pine Surfel GPU storage buffers
        let dummy_proto = [SplatProto::zeroed()];
        let dummy_place = [SplatPlacement::zeroed()];
        let dummy_ref = [SplatRef::zeroed()];
        let proto_slice: &[SplatProto] = if pines.protos.is_empty() { &dummy_proto } else { &pines.protos };
        let place_slice: &[SplatPlacement] = if pines.placements.is_empty() { &dummy_place } else { &pines.placements };
        let ref_slice: &[SplatRef] = if pines.draw_refs.is_empty() { &dummy_ref } else { &pines.draw_refs };

        let surfel_proto_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pine_surfel_protos"),
            size: (proto_slice.len() * std::mem::size_of::<SplatProto>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surfel_place_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pine_surfel_placements"),
            size: (place_slice.len() * std::mem::size_of::<SplatPlacement>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surfel_ref_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pine_surfel_refs"),
            size: (ref_slice.len() * std::mem::size_of::<SplatRef>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&surfel_proto_buf, 0, bytemuck::cast_slice(proto_slice));
        queue.write_buffer(&surfel_place_buf, 0, bytemuck::cast_slice(place_slice));
        queue.write_buffer(&surfel_ref_buf, 0, bytemuck::cast_slice(ref_slice));

        let bg_surfels = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_surfels"),
            layout: &surfel_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: surfel_proto_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: surfel_place_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: surfel_ref_buf.as_entire_binding() },
            ],
        });

        // Compile WGSL shader modules
        let raymarch_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_raymarch"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/volumetric_raymarch.wgsl"))),
        });
        let field_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_mesh"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/field_mesh.wgsl"))),
        });
        let pine_surfel_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pine_surfels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/pine_surfels.wgsl"))),
        });
        let overlay_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_overlay"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/volumetric_overlay.wgsl"))),
        });
        let resolve_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel_resolve"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/pixel_resolve.wgsl"))),
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
        let surfel_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surfel_pl"),
            bind_group_layouts: &[Some(&vol_bgl), Some(&surfel_bgl)],
            immediate_size: 0,
        });
        let overlay_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_pl"),
            bind_group_layouts: &[Some(&overlay_bgl)],
            immediate_size: 0,
        });
        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resolve_pl"),
            bind_group_layouts: &[Some(&resolve_bgl)],
            immediate_size: 0,
        });

        // Pipeline 1: Renderer A (Raymarch fullscreen)
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

        // Pipeline 2: Renderer B Pass 1a (Surface-Nets 3D solid mesh -> linear HDR + depth)
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

        // Pipeline 2b: Hard-Z GPU-Instanced Sculpted Pine Surfels -> linear HDR + hardware depth!
        let pipe_pine_surfels = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_pine_surfels"),
            layout: Some(&surfel_pl),
            vertex: wgpu::VertexState {
                module: &pine_surfel_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &pine_surfel_mod,
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

        // Pipeline 3: Renderer B Pass 2 (Volumetric Fluid/Gas Overlay -> resolve_src Rgba8Unorm!)
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

        // Pipeline 4: Renderer B Pass 3 (Stage A 2xSS HD Pixel-Art Resolve / Clean-3D Toggle -> out_format!)
        let pipe_pixel_resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_pixel_resolve"),
            layout: Some(&resolve_pl),
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

        // Pipelines 5 & 6: Shared 3D Hero Knights
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

        let (depth_view, hdr_view, resolve_src_view) = make_targets(&device, 1280, 720);
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
        let bg_resolve = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_resolve"),
            layout: &resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: resolve_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&resolve_src_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&depth_view) },
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
            visual_style: VisualStyle::NordicPixel2xSS,
            resolve_style: ResolveStyle::Pixel2xSS,
            neural_decoder: None,
            scene_ubuf_a,
            scene_ubuf_b,
            resolve_ubuf,
            _density_tex: density_tex,
            albedo_tex,
            _brick_tex: brick_tex,
            density_view,
            albedo_view,
            brick_view,
            vol_samp,
            pipe_raymarch,
            pipe_field_mesh,
            pipe_pine_surfels,
            pipe_vol_overlay,
            pipe_pixel_resolve,
            pipe_hero_a,
            pipe_hero_b,
            bg_vol_a,
            bg_vol_b,
            bg_surfels,
            overlay_bgl,
            resolve_bgl,
            sn_vbuf,
            sn_ibuf,
            sn_index_count: sn_idxs.len() as u32,
            sn_vertex_count: sn_verts.len() as u32,
            _surfel_proto_buf: surfel_proto_buf,
            _surfel_place_buf: surfel_place_buf,
            _surfel_ref_buf: surfel_ref_buf,
            surfel_draw_count: pines.draw_refs.len() as u32,
            pine_count: pines.pine_count,
            snow_pine_count: pines.snow_pine_count,
            hero_vbuf,
            hero_ibuf,
            hero_index_count: hero_mesh.indices.len() as u32,
            hero_inst_buf,
            hero_inst_count: hero_instances.len() as u32,
            target_w: 1280,
            target_h: 720,
            depth_view,
            hdr_view,
            resolve_src_view,
            bg_overlay,
            bg_resolve,
        })
    }

    /// Attach the Client-Side WGSL `NeuralMaterialHead(Z_world)` Compute Decoder directly to this
    /// renderer's live 3D `density_tex` & `albedo_tex` and immediately GPU-decode Style 0!
    pub fn attach_neural_decoder(
        &mut self,
        packet: &LatentBrickPacket,
        weights: &ClientMaterialWeights,
    ) {
        let dec = GpuNeuralDecoder::new(
            &self.device,
            &self.queue,
            packet,
            weights,
            &self.density_view,
            &self.albedo_view,
        );
        dec.decode_full(&self.device, &self.queue, self.visual_style);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.neural_decoder = Some(dec);
    }

    /// Live-swap the Client-Side Visual Style (`Y`-key!) by re-decoding the exact same
    /// `Z_world_3d` latent bricks on the GPU in `< 1 ms`!
    pub fn set_visual_style(&mut self, style: VisualStyle) -> f64 {
        self.visual_style = style;
        if let Some(ref dec) = self.neural_decoder {
            let t0 = Instant::now();
            dec.decode_full(&self.device, &self.queue, style);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            return t0.elapsed().as_secs_f64() * 1000.0;
        }
        0.0
    }

    /// Benchmark full-world (1,048,576 voxels) and single 16×16×16 brick WGSL neural decode
    /// AND verify bit-level parity against CPU reference decode!
    pub fn benchmark_neural_decode(
        &mut self,
        cpu_ref_vol: &WorldVolume3D,
        style: VisualStyle,
        warmup: usize,
        iters: usize,
    ) -> Option<GpuDecodeBenchMetrics> {
        let dec = self.neural_decoder.as_ref()?;
        let (full_p50, full_p99, full_mean, brick_p50, brick_p99, brick_mean) =
            dec.benchmark(&self.device, &self.queue, style, warmup, iters);

        // Read back one Z=128 slice of albedo_tex to verify exact GPU-vs-CPU parity!
        let w = self.vol_dims[0] as u32;
        let d = self.vol_dims[1] as u32;
        let row_bytes = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let slice_z = (self.vol_dims[2] as u32 / 2).min(self.vol_dims[2] as u32 - 1);

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parity_readback"),
            size: (padded * d) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.albedo_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: slice_z },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(d),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: d,
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

        let mut max_diff = 0u8;
        let mut sum_diff = 0u64;
        let mut count = 0u64;
        for y in 0..d as usize {
            let row_off = y * padded as usize;
            for x in 0..w as usize {
                let cpu_i = cpu_ref_vol.byte_index(x, y, slice_z as usize);
                for c in 0..4 {
                    let g = mapped[row_off + x * 4 + c];
                    let r = cpu_ref_vol.albedo_thermal[cpu_i + c];
                    let diff = g.abs_diff(r);
                    max_diff = max_diff.max(diff);
                    sum_diff += diff as u64;
                    count += 1;
                }
            }
        }
        drop(mapped);
        staging.unmap();

        Some(GpuDecodeBenchMetrics {
            decode_full_1m_vox_p50_ms: full_p50,
            decode_full_1m_vox_p99_ms: full_p99,
            decode_full_1m_vox_mean_ms: full_mean,
            decode_single_16x16x16_brick_p50_ms: brick_p50,
            decode_single_16x16x16_brick_p99_ms: brick_p99,
            decode_single_16x16x16_brick_mean_ms: brick_mean,
            gpu_vs_cpu_max_u8_diff: max_diff,
            gpu_vs_cpu_mae_u8: sum_diff as f64 / count.max(1) as f64,
        })
    }

    fn ensure_targets(&mut self, w: u32, h: u32) {
        if self.target_w == w && self.target_h == h {
            return;
        }
        let (depth_view, hdr_view, resolve_src_view) = make_targets(&self.device, w, h);
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
        self.bg_resolve = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_resolve"),
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.resolve_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&resolve_src_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&depth_view) },
            ],
        });
        self.depth_view = depth_view;
        self.hdr_view = hdr_view;
        self.resolve_src_view = resolve_src_view;
        self.target_w = w;
        self.target_h = h;
    }

    fn build_uniforms(&self, cam: &Camera3D, w: u32, h: u32, sim_time: f32, tonemap: f32) -> SceneUniforms {
        let aspect = w.max(1) as f32 / h.max(1) as f32;
        let mats = cam.matrices(aspect, h as f32);
        let sun_dir = Vec3::new(0.56, 0.54, 0.63).normalize();
        let (sun_col, sky_top, sky_horizon, exposure, _) = self.visual_style.atmosphere();
        let stylized = match self.resolve_style {
            ResolveStyle::Pixel2xSS => 1.0,
            ResolveStyle::Clean3D => 0.0,
        };
        SceneUniforms {
            view_proj: mats.view_proj.to_cols_array_2d(),
            inv_view_proj: mats.inv_view_proj.to_cols_array_2d(),
            view: mats.view.to_cols_array_2d(),
            proj: mats.proj.to_cols_array_2d(),
            cam_pos: [cam.pos.x, cam.pos.y, cam.pos.z, sim_time],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
            sun_col: [sun_col[0], sun_col[1], sun_col[2], mats.focal_px],
            sky_top: [sky_top[0], sky_top[1], sky_top[2], self.visual_style.id() as f32],
            sky_horizon: [sky_horizon[0], sky_horizon[1], sky_horizon[2], stylized],
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

    fn build_resolve_uniforms(&self, cam: &Camera3D, w: u32, h: u32) -> ResolveUniforms {
        let (pal, pal_n) = self.visual_style.resolve_palette_64();
        let (_, _, _, _, ink_tint) = self.visual_style.atmosphere();
        let is_pixel = self.resolve_style == ResolveStyle::Pixel2xSS;
        ResolveUniforms {
            out_res: [w as f32, h as f32],
            cell_scale: 2.0,
            outline: if is_pixel { 1.0 } else { 0.0 },
            quantise: if is_pixel { 1.0 } else { 0.0 },
            dither: 0.24,
            crisp: if is_pixel { 1.0 } else { 0.0 },
            palette_count: pal_n,
            depth_params: [cam.near, cam.far, self.visual_style.id() as f32, 0.0],
            ink_tint,
            palette: pal,
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
                let res_uni = self.build_resolve_uniforms(cam, w, h);
                self.queue.write_buffer(&self.scene_ubuf_b, 0, bytemuck::bytes_of(&uni));
                self.queue.write_buffer(&self.resolve_ubuf, 0, bytemuck::bytes_of(&res_uni));

                // Pass 1: Smooth Surface-Nets + Hard-Z Sculpted Pine Surfels + 3D Hero Knights -> HDR + Depth
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("surface_nets_pines_heroes"),
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

                    if self.surfel_draw_count > 0 {
                        rp.set_pipeline(&self.pipe_pine_surfels);
                        rp.set_bind_group(0, &self.bg_vol_b, &[]);
                        rp.set_bind_group(1, &self.bg_surfels, &[]);
                        rp.draw(0..6, 0..self.surfel_draw_count);
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

                // Pass 2: Depth-Tested Volumetric Fluid & Gas Overlay + Sky + ACES Tonemap -> resolve_src
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("volumetric_overlay"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.resolve_src_view,
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

                // Pass 3: Stage A 2xSS HD Pixel-Art Resolve (or Clean-3D when T-key toggled!) -> out_view
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stage_a_pixel_resolve"),
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

                    rp.set_pipeline(&self.pipe_pixel_resolve);
                    rp.set_bind_group(0, &self.bg_resolve, &[]);
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
