//! Iteration 9A: Client-Side WGSL `NeuralMaterialHead(Z_world)` Compute Decoder,
//! `FMLT` Compact Latent Brick Packet Loader, `FMWG` Swappable Style Weight Loader,
//! and Per-Style Visual Atmosphere / 64-Color Resolve Palettes!

use crate::volume::WorldVolume3D;
use bytemuck::{Pod, Zeroable};
use serde::Serialize;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

pub const FMLT_MAGIC: &[u8; 4] = b"FMLT";
pub const FMWG_MAGIC: &[u8; 4] = b"FMWG";
pub const IN_DIM: usize = 8;
pub const H1_DIM: usize = 24;
pub const H2_DIM: usize = 24;
pub const OUT_DIM: usize = 8;
pub const FLOATS_PER_STYLE: usize =
    H1_DIM * IN_DIM + H1_DIM + H2_DIM * H1_DIM + H2_DIM + OUT_DIM * H2_DIM + OUT_DIM;
pub const NUM_STYLES: usize = 3;

/// Baked 12 KB fallback weights (compiled directly from `models/checkpoints/client_material_weights.bin`)
/// so the client works out-of-the-box even without an explicit `--weights` CLI path!
const BAKED_WEIGHTS_FMWG: &[u8] =
    include_bytes!("../../../models/checkpoints/client_material_weights.bin");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisualStyle {
    NordicPixel2xSS = 0,
    VolcanicEmber = 1,
    ArcaneThermal = 2,
}

impl VisualStyle {
    pub fn id(self) -> u32 {
        self as u32
    }

    pub fn next(self) -> Self {
        match self {
            VisualStyle::NordicPixel2xSS => VisualStyle::VolcanicEmber,
            VisualStyle::VolcanicEmber => VisualStyle::ArcaneThermal,
            VisualStyle::ArcaneThermal => VisualStyle::NordicPixel2xSS,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            VisualStyle::NordicPixel2xSS => "NORDIC PIXEL-2XSS",
            VisualStyle::VolcanicEmber => "VOLCANIC EMBER",
            VisualStyle::ArcaneThermal => "ARCANE THERMAL",
        }
    }

    /// Atmospheric lighting preset paired with each decoded neural material style:
    /// `(sun_col, sky_top, sky_horizon, exposure, ink_tint_rgba)`.
    pub fn atmosphere(self) -> ([f32; 3], [f32; 3], [f32; 3], f32, [f32; 4]) {
        match self {
            VisualStyle::NordicPixel2xSS => (
                [1.96, 1.72, 1.32],
                [0.06, 0.21, 0.56],
                [0.42, 0.62, 0.84],
                1.14,
                [0.32, 0.30, 0.48, 0.68], // Cool-Indigo #261e48 ink outline
            ),
            VisualStyle::VolcanicEmber => (
                [2.15, 1.05, 0.52],
                [0.14, 0.05, 0.09],
                [0.52, 0.20, 0.12],
                1.14,
                [0.28, 0.16, 0.22, 0.72], // Obsidian-crimson ink outline
            ),
            VisualStyle::ArcaneThermal => (
                [0.95, 1.48, 1.95],
                [0.02, 0.06, 0.22],
                [0.08, 0.22, 0.44],
                1.18,
                [0.16, 0.22, 0.52, 0.70], // Deep arcane-indigo ink outline
            ),
        }
    }

    /// Style-harmonized 64-color palette for Stage A 2xSS `pixel_resolve.wgsl`.
    pub fn resolve_palette_64(self) -> ([[f32; 4]; 64], f32) {
        let hexes: &[u32] = match self {
            VisualStyle::NordicPixel2xSS => &[
                0x161430, 0x261e48, 0x2f3b5c, 0x48587a, 0x65799e, 0x89a0c0, 0xb6c9e0, 0xe6eff8,
                0x112a26, 0x1c4435, 0x2c6942, 0x428d4d, 0x6bb453, 0x9cd464, 0xd0eb86, 0x2a1a1c,
                0x452b25, 0x6b412f, 0x98613f, 0xc48b58, 0xe2be8f, 0xfff4d6, 0x821e30, 0xba3138,
                0xe55242, 0xf47e38, 0xf9b536, 0xfff078, 0xfffbe0, 0x10355d, 0x1d5f94, 0x308ac4,
                0x5cc4ec, 0xa8f2ff, 0x2b193d, 0x4e2d6e, 0x7e4c9e, 0x1f2028, 0x403e4a, 0xa17528,
                0xe6b942, 0x5c241e, 0x893928, 0xb65438, 0xd98050, 0x698eb2, 0x9bc0dc, 0xcfe6f5,
                0xf2faff, 0x1d3e55, 0x396a84, 0x67a2ba, 0x4c5e2c, 0x7a9442, 0x22608a, 0x348be0,
                0x6ec3ff, 0x6d615a, 0x9b8d82, 0x5c728a, 0x3a4d66, 0x7898b8, 0xc8d8e8,
            ],
            VisualStyle::VolcanicEmber => &[
                0x120c14, 0x1d141e, 0x2c1e26, 0x3e2a30, 0x543a3c, 0x6e4e4a, 0x8d685c, 0xb38a74,
                0xd8b08a, 0xf2d4a6, 0x281012, 0x441616, 0x681d18, 0x92271a, 0xc2381c, 0xe8541e,
                0xff7c24, 0xffaa32, 0xffd858, 0xfff6a8, 0xfffce0, 0x1a151c, 0x30242c, 0x4b3539,
                0x694744, 0x8a5b4e, 0xad7358, 0xd19066, 0x4a1e14, 0x76301b, 0xa44622, 0xd6642a,
                0xf48c38, 0xffba4c, 0x3a1c28, 0x5a2834, 0x7e383e, 0xa44e44, 0xc86b4c, 0xe8905e,
                0x241818, 0x3c2824, 0x583a30, 0x785040, 0x9c6c52, 0xc48e6a, 0xe6b488, 0xf9dcb0,
            ],
            VisualStyle::ArcaneThermal => &[
                0x070a1c, 0x0e1432, 0x16204c, 0x1f2f68, 0x2a4286, 0x3a5aa4, 0x527ac2, 0x78a4de,
                0x082238, 0x0a3c5c, 0x0e5c82, 0x1482aa, 0x1caec8, 0x38d8e8, 0x76f2ff, 0xcafcff,
                0x1c1240, 0x341c64, 0x542488, 0x7c28a6, 0xa828b8, 0xd42ac0, 0xf638b2, 0xff6298,
                0xff8e74, 0xffc058, 0xffe880, 0xfffad0, 0x082b34, 0x0e4d52, 0x167472, 0x20a092,
                0x36cbb2, 0x6ef0d2, 0xb8fff0, 0x12183a, 0x202a5e, 0x324086, 0x4a5cae, 0x6a80d0,
                0x92aaf0, 0xc0d6ff, 0xe6f2ff, 0x30144c, 0x601c78, 0x9a2496, 0xd8329c, 0xff5888,
            ],
        };
        let mut out = [[0.0f32; 4]; 64];
        for (i, &hex) in hexes.iter().take(64).enumerate() {
            out[i] = [
                ((hex >> 16) & 0xff) as f32 / 255.0,
                ((hex >> 8) & 0xff) as f32 / 255.0,
                (hex & 0xff) as f32 / 255.0,
                1.0,
            ];
        }
        (out, hexes.len().min(64) as f32)
    }
}

/// Compact Server-to-Client Latent Brick Packet (`FMLT`, zero baked RGB!).
/// Stores `[depth=16, height=256, width=256, 8]` quantized `u8` features:
/// `(pca0, pca1, pca2, pca3, solid_density, fluid_density, gas_density, temp_norm)`.
pub struct LatentBrickPacket {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub channels: usize,
    pub seed: u32,
    /// Raw Z-major `[alt][north][east][8]` bytes (directly uploadable as `array<vec2<u32>>`!).
    pub raw_u8: Vec<u8>,
}

impl LatentBrickPacket {
    pub fn load_fmlt(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != FMLT_MAGIC {
            return Err(format!("Invalid FMLT magic: {:?}", magic).into());
        }
        let mut hdr = [0u8; 20];
        r.read_exact(&mut hdr)?;
        let width = u32::from_le_bytes(hdr[0..4].try_into()?) as usize;
        let height = u32::from_le_bytes(hdr[4..8].try_into()?) as usize;
        let depth = u32::from_le_bytes(hdr[8..12].try_into()?) as usize;
        let channels = u32::from_le_bytes(hdr[12..16].try_into()?) as usize;
        let seed = u32::from_le_bytes(hdr[16..20].try_into()?);
        if channels != IN_DIM {
            return Err(format!("Expected FMLT channels={}, got {}", IN_DIM, channels).into());
        }
        let n = width * height * depth * channels;
        let mut raw_u8 = vec![0u8; n];
        r.read_exact(&mut raw_u8)?;
        Ok(Self {
            width,
            height,
            depth,
            channels,
            seed,
            raw_u8,
        })
    }

    pub fn voxel_count(&self) -> usize {
        self.width * self.height * self.depth
    }

    pub fn byte_size(&self) -> usize {
        24 + self.raw_u8.len()
    }
}

/// 3 Swappable Client-Side WGSL `NeuralMaterialHead` Weight Sets (`FMWG`, ~4 KB per style!).
pub struct ClientMaterialWeights {
    pub num_styles: usize,
    pub weights_f32: Vec<f32>,
}

impl ClientMaterialWeights {
    pub fn from_bytes(raw: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if raw.len() < 24 || &raw[0..4] != FMWG_MAGIC {
            return Err("Invalid FMWG weights header".into());
        }
        let num_styles = u32::from_le_bytes(raw[4..8].try_into()?) as usize;
        let in_dim = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let h1_dim = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        let h2_dim = u32::from_le_bytes(raw[16..20].try_into()?) as usize;
        let out_dim = u32::from_le_bytes(raw[20..24].try_into()?) as usize;
        if (in_dim, h1_dim, h2_dim, out_dim) != (IN_DIM, H1_DIM, H2_DIM, OUT_DIM) {
            return Err(format!(
                "FMWG arch mismatch: got ({in_dim},{h1_dim},{h2_dim},{out_dim})"
            )
            .into());
        }
        let expected_floats = num_styles * FLOATS_PER_STYLE;
        let expected_bytes = 24 + expected_floats * 4;
        if raw.len() != expected_bytes {
            return Err(format!(
                "FMWG byte length {} != expected {}",
                raw.len(),
                expected_bytes
            )
            .into());
        }
        let mut weights_f32 = Vec::with_capacity(expected_floats);
        for chunk in raw[24..].chunks_exact(4) {
            weights_f32.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Self {
            num_styles,
            weights_f32,
        })
    }

    pub fn load_or_baked(path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        if let Some(p) = path {
            if p.exists() {
                let bytes = std::fs::read(p)?;
                return Self::from_bytes(&bytes);
            }
        }
        Self::from_bytes(BAKED_WEIGHTS_FMWG)
    }

    /// Exact CPU reference evaluation of one voxel (matches `neural_material_decode.wgsl` bit-for-bit!).
    #[inline]
    pub fn eval_voxel(&self, style_id: usize, x_u8: &[u8; 8]) -> [f32; 8] {
        let base = style_id * FLOATS_PER_STYLE;
        let w = &self.weights_f32[base..base + FLOATS_PER_STYLE];
        let mut x = [0.0f32; 8];
        for i in 0..8 {
            x[i] = x_u8[i] as f32 / 255.0;
        }
        let mut h1 = [0.0f32; H1_DIM];
        for o in 0..H1_DIM {
            let mut acc = w[192 + o];
            let row = o * 8;
            for i in 0..8 {
                acc += w[row + i] * x[i];
            }
            h1[o] = acc / (1.0 + (-acc).exp());
        }
        let mut h2 = [0.0f32; H2_DIM];
        for o in 0..H2_DIM {
            let mut acc = w[792 + o];
            let row = 216 + o * H1_DIM;
            for i in 0..H1_DIM {
                acc += w[row + i] * h1[i];
            }
            h2[o] = acc / (1.0 + (-acc).exp()) + h1[o];
        }
        let mut out = [0.0f32; OUT_DIM];
        for o in 0..OUT_DIM {
            let mut acc = w[1008 + o];
            let row = 816 + o * H2_DIM;
            for i in 0..H2_DIM {
                acc += w[row + i] * h2[i];
            }
            out[o] = 1.0 / (1.0 + (-acc).exp());
        }
        out
    }

    /// CPU reference decode of an entire `LatentBrickPacket` into `WorldVolume3D` (used for
    /// initial CPU Surface-Nets mesh extraction, pine placement, brick occupancy, and GPU parity tests!).
    pub fn decode_to_volume_cpu(&self, packet: &LatentBrickPacket, style: VisualStyle) -> WorldVolume3D {
        let mut vol = WorldVolume3D::new_empty(packet.width, packet.depth, packet.height);
        let style_id = (style.id() as usize).min(self.num_styles.saturating_sub(1));
        for alt in 0..packet.depth {
            for north in 0..packet.height {
                for east in 0..packet.width {
                    let src = ((alt * packet.height + north) * packet.width + east) * 8;
                    let mut feat = [0u8; 8];
                    feat.copy_from_slice(&packet.raw_u8[src..src + 8]);
                    let y = self.eval_voxel(style_id, &feat);
                    let dst = vol.byte_index(east, alt, north);
                    for c in 0..4 {
                        vol.density_mat[dst + c] = (y[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        vol.albedo_thermal[dst + c] = (y[4 + c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
            }
        }
        vol
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DecodeParams {
    vol_dims: [u32; 4],
    region_min: [u32; 4],
    region_size: [u32; 4],
}

#[derive(Serialize, Clone, Debug)]
pub struct GpuDecodeBenchMetrics {
    pub decode_full_1m_vox_p50_ms: f64,
    pub decode_full_1m_vox_p99_ms: f64,
    pub decode_full_1m_vox_mean_ms: f64,
    pub decode_single_16x16x16_brick_p50_ms: f64,
    pub decode_single_16x16x16_brick_p99_ms: f64,
    pub decode_single_16x16x16_brick_mean_ms: f64,
    pub gpu_vs_cpu_max_u8_diff: u8,
    pub gpu_vs_cpu_mae_u8: f64,
}

/// GPU Compute Pipeline executing `shaders/neural_material_decode.wgsl` directly into the
/// renderer's live `Rgba8Unorm` 3D textures (`density_mat` + `albedo_thermal`)!
pub struct GpuNeuralDecoder {
    vol_dims: [u32; 3],
    params_ubuf: wgpu::Buffer,
    _latent_buf: wgpu::Buffer,
    _weights_buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl GpuNeuralDecoder {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &LatentBrickPacket,
        weights: &ClientMaterialWeights,
        density_storage_view: &wgpu::TextureView,
        albedo_storage_view: &wgpu::TextureView,
    ) -> Self {
        let params_ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neural_decode_params"),
            size: std::mem::size_of::<DecodeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let latent_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fmlt_latent_bricks_storage"),
            size: packet.raw_u8.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&latent_buf, 0, &packet.raw_u8);

        let weights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fmwg_style_weights_storage"),
            size: (weights.weights_f32.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&weights_buf, 0, bytemuck::cast_slice(&weights.weights_f32));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("neural_decode_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("neural_decode_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: latent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(density_storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(albedo_storage_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neural_material_decode_wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/neural_material_decode.wgsl"
            ))),
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neural_decode_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipe_neural_material_decode"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_decode"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            vol_dims: [packet.width as u32, packet.depth as u32, packet.height as u32],
            params_ubuf,
            _latent_buf: latent_buf,
            _weights_buf: weights_buf,
            pipeline,
            bind_group,
        }
    }

    pub fn dispatch_region(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        style: VisualStyle,
        origin: [u32; 3],
        size: [u32; 3],
    ) {
        let params = DecodeParams {
            vol_dims: [self.vol_dims[0], self.vol_dims[1], self.vol_dims[2], style.id()],
            region_min: [origin[0], origin[1], origin[2], 0],
            region_size: [size[0], size[1], size[2], 0],
        };
        queue.write_buffer(&self.params_ubuf, 0, bytemuck::bytes_of(&params));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("neural_decode_enc"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("neural_decode_pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &self.bind_group, &[]);
            cp.dispatch_workgroups(
                size[0].div_ceil(8),
                size[1].div_ceil(4),
                size[2].div_ceil(8),
            );
        }
        queue.submit(Some(enc.finish()));
    }

    /// Decode all 1,048,576 voxels (`256×16×256`) directly into the live 3D textures on GPU!
    pub fn decode_full(&self, device: &wgpu::Device, queue: &wgpu::Queue, style: VisualStyle) {
        self.dispatch_region(device, queue, style, [0, 0, 0], self.vol_dims);
    }

    /// Decode a single dirty `16×16×16` brick directly into the live 3D textures on GPU!
    pub fn decode_single_brick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        style: VisualStyle,
        bx: u32,
        bz: u32,
    ) {
        let ox = (bx * 16).min(self.vol_dims[0].saturating_sub(16));
        let oz = (bz * 16).min(self.vol_dims[2].saturating_sub(16));
        self.dispatch_region(
            device,
            queue,
            style,
            [ox, 0, oz],
            [16, self.vol_dims[1].min(16), 16],
        );
    }

    /// Benchmark full 1M-voxel GPU decode latency and single 16×16×16 dirty-brick GPU decode latency.
    /// Uses amortized batched submits inside each measurement window to measure pure GPU execution
    /// without per-dispatch CPU poll overhead inflating sub-0.05ms brick timings!
    pub fn benchmark(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        style: VisualStyle,
        warmup: usize,
        iters: usize,
    ) -> (f64, f64, f64, f64, f64, f64) {
        // 1. Full 1,048,576-voxel decode benchmark
        for _ in 0..warmup {
            self.decode_full(device, queue, style);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let mut full_samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            self.decode_full(device, queue, style);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            full_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        full_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let full_p50 = full_samples[full_samples.len() / 2];
        let full_p99 = full_samples[((full_samples.len() as f64) * 0.99)
            .floor()
            .min((full_samples.len() - 1) as f64) as usize];
        let full_mean = full_samples.iter().sum::<f64>() / full_samples.len().max(1) as f64;

        // 2. Single 16x16x16 brick decode benchmark (batched 32 dispatches per encoder to avoid poll floor!)
        let batch_n = 32u32;
        let params = DecodeParams {
            vol_dims: [self.vol_dims[0], self.vol_dims[1], self.vol_dims[2], style.id()],
            region_min: [128, 0, 128, 0],
            region_size: [16, self.vol_dims[1].min(16), 16, 0],
        };
        queue.write_buffer(&self.params_ubuf, 0, bytemuck::bytes_of(&params));

        for _ in 0..warmup {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.pipeline);
                cp.set_bind_group(0, &self.bind_group, &[]);
                for _ in 0..batch_n {
                    cp.dispatch_workgroups(2, 4, 2);
                }
            }
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        let mut brick_samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.pipeline);
                cp.set_bind_group(0, &self.bind_group, &[]);
                for _ in 0..batch_n {
                    cp.dispatch_workgroups(2, 4, 2);
                }
            }
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            brick_samples.push(t0.elapsed().as_secs_f64() * 1000.0 / batch_n as f64);
        }
        brick_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let brick_p50 = brick_samples[brick_samples.len() / 2];
        let brick_p99 = brick_samples[((brick_samples.len() as f64) * 0.99)
            .floor()
            .min((brick_samples.len() - 1) as f64) as usize];
        let brick_mean = brick_samples.iter().sum::<f64>() / brick_samples.len().max(1) as f64;

        // Restore full decoded volume for active style
        self.decode_full(device, queue, style);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        (full_p50, full_p99, full_mean, brick_p50, brick_p99, brick_mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baked_weights_load_and_eval() {
        let weights = ClientMaterialWeights::load_or_baked(None).expect("load baked FMWG");
        assert_eq!(weights.num_styles, 3);
        assert_eq!(weights.weights_f32.len(), 3 * FLOATS_PER_STYLE);
        let feat = [128u8, 128, 128, 128, 250, 0, 0, 20];
        let y0 = weights.eval_voxel(0, &feat);
        let y1 = weights.eval_voxel(1, &feat);
        let y2 = weights.eval_voxel(2, &feat);
        for v in y0.iter().chain(y1.iter()).chain(y2.iter()) {
            assert!(*v >= 0.0 && *v <= 1.0 && v.is_finite());
        }
        // Styles should produce distinct albedos on identical latent input!
        let d01 = (y0[4] - y1[4]).abs() + (y0[5] - y1[5]).abs() + (y0[6] - y1[6]).abs();
        let d02 = (y0[4] - y2[4]).abs() + (y0[5] - y2[5]).abs() + (y0[6] - y2[6]).abs();
        assert!(d01 > 0.02 || d02 > 0.02, "Swapped styles must diverge on same latent");
    }
}
