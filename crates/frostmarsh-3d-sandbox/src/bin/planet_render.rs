//! Wave 29B: Curved Apple M4 Pro Metal Planetary Renderer (`planet_render`).
//!
//! Instantiates a live 6-face `CubedSpherePlanet` (`64×64×32×6 = 786,432` active 3D voxels,
//! `r_core = 48.0m`, outer radius `80.0m`), runs 10 conservation-exact planetary physics +
//! sweeping solar terminator ticks (asserting `global_mass_error() == 0 && global_energy_error() == 0`),
//! extracts seamless curved 3D spherical meshes (terrain, Fresnel ocean, and Orbital Floating
//! Sky-Ring Islands with glowing Buoyant Crystal Cores), renders with physical depth-aware
//! Rayleigh+Mie atmospheric limb scattering & cosmic starfield on Apple M4 Pro Metal,
//! benchmarks 1080p & 4K FPS (25 iters), and writes:
//!   - `models/planet_showcase/planet_orbital_globe_1080p.png`
//!   - `models/planet_showcase/planet_horizon_surface_1080p.png`
//!   - `models/planet_showcase/planet_4view_showcase.png`
//!   - `models/PLANET_METAL_METRICS.json`

use bytemuck::{Pod, Zeroable};
use field_core::planet::{
    CubedSpherePlanet, FACE_COUNT, FACE_NEG_X, FACE_NEG_Z, FACE_POS_X, FACE_POS_Z,
    face_uv_to_sphere_dir, face_uv_z_to_world_pos, sphere_dir_to_face_uv,
};
use field_core::{SP_ICE, SP_OIL, SP_PLANT, SP_ROCK, SP_SMOKE, SP_SOIL, SP_STEAM, SP_WATER, SP_WOOD};
use glam::{Mat4, Vec3};
use image::{Rgba, RgbaImage};
use serde::Serialize;
use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

// =============================================================================
// Vertex & Uniform Layouts
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlanetVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    emissive: [f32; 3],
    roughness: f32,
    metallic: f32,
    water_depth: f32,
    ice_frac: f32,
}

impl PlanetVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PlanetVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlanetUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],    // xyz, time
    sun_dir: [f32; 4],    // xyz, r_ocean
    planet_cfg: [f32; 4], // r_core, r_surf_atmo, r_atmo_outer, exposure
    viewport: [f32; 4],   // width, height, 1/width, 1/height
}

struct CameraShot {
    pos: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y_deg: f32,
    near: f32,
    far: f32,
    sun_dir: Vec3,
    exposure: f32,
}

impl CameraShot {
    fn view_proj(&self, w: u32, h: u32) -> (Mat4, Mat4) {
        let aspect = (w.max(1) as f32) / (h.max(1) as f32);
        let view = glam::camera::rh::view::look_at_mat4(self.pos, self.target, self.up);
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fov_y_deg.to_radians(),
            aspect,
            self.near,
            self.far,
        );
        let vp = proj * view;
        (vp, vp.inverse())
    }
}

// =============================================================================
// WGSL Shaders (Starfield Backdrop, Curved Terrain/Sky-Ring, Fresnel Ocean,
//               Physical Depth-Aware Rayleigh+Mie Atmosphere + ACES Tonemap)
// =============================================================================

const PLANET_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    planet_cfg: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

// -----------------------------------------------------------------------------
// Hash & Procedural Noise Helpers (3D seamless spherical micro-detail)
// -----------------------------------------------------------------------------
fn hash31(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        v += a * noise3(q);
        q = q * 2.07 + vec3<f32>(13.1, -7.4, 19.7);
        a *= 0.5;
    }
    return v;
}

// -----------------------------------------------------------------------------
// Full-Screen Cosmic Starfield Backdrop Pass (depth = 1.0)
// -----------------------------------------------------------------------------
struct FullScreenOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullScreenOut {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    var out: FullScreenOut;
    let p = pts[vi];
    out.pos = vec4<f32>(p, 0.99999, 1.0);
    out.ndc = p;
    return out;
}

fn ray_dir_from_ndc(ndc: vec2<f32>) -> vec3<f32> {
    let p_near = uni.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let p_far = uni.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let wn = p_near.xyz / p_near.w;
    let wf = p_far.xyz / p_far.w;
    return normalize(wf - wn);
}

@fragment
fn fs_starfield(in: FullScreenOut) -> @location(0) vec4<f32> {
    let rd = ray_dir_from_ndc(in.ndc);
    let sun = normalize(uni.sun_dir.xyz);

    // Deep cosmic velvet background + subtle galactic dust band
    var col = vec3<f32>(0.003, 0.005, 0.012);
    let band = exp(-pow(dot(rd, normalize(vec3<f32>(0.28, 0.85, -0.44))) * 4.2, 2.0));
    let neb = fbm3(rd * 4.5 + vec3<f32>(2.1, 0.7, -1.3));
    col += band * vec3<f32>(0.018, 0.026, 0.052) * smoothstep(0.32, 0.82, neb);
    col += band * vec3<f32>(0.024, 0.012, 0.036) * smoothstep(0.55, 0.90, fbm3(rd * 8.0 + 4.0));

    // Crisp multi-scale pinpoint stars (2 octaves on unit sphere)
    let g1 = rd * 240.0;
    let c1 = floor(g1);
    let f1 = fract(g1) - 0.5;
    let h1 = hash31(c1);
    if (h1 > 0.992) {
        let jitter = vec3<f32>(hash31(c1 + 7.1), hash31(c1 + 19.3), hash31(c1 + 37.9)) - 0.5;
        let d = length(f1 - jitter * 0.55);
        let b = smoothstep(0.32, 0.04, d) * (0.55 + 1.45 * fract(h1 * 137.0));
        let tint = mix(vec3<f32>(0.72, 0.84, 1.0), vec3<f32>(1.0, 0.90, 0.72), fract(h1 * 53.0));
        col += tint * b * 1.4;
    }

    let g2 = rd * 135.0;
    let c2 = floor(g2);
    let f2 = fract(g2) - 0.5;
    let h2 = hash31(c2 + 113.0);
    if (h2 > 0.9965) {
        let jitter2 = vec3<f32>(hash31(c2 + 11.7), hash31(c2 + 29.1), hash31(c2 + 43.3)) - 0.5;
        let d2 = length(f2 - jitter2 * 0.45);
        let b2 = smoothstep(0.36, 0.03, d2) * (1.2 + 2.2 * fract(h2 * 91.0));
        let tint2 = mix(vec3<f32>(0.68, 0.86, 1.0), vec3<f32>(1.0, 0.82, 0.62), fract(h2 * 29.0));
        col += tint2 * b2 * 1.8;
    }

    // Distant solar disc + soft corona glow when sun is in camera field
    let sun_cos = max(dot(rd, sun), 0.0);
    let sun_disc = smoothstep(0.9993, 0.99985, sun_cos);
    let sun_halo = pow(sun_cos, 48.0) * 0.45 + pow(sun_cos, 6.0) * 0.03;
    col += vec3<f32>(1.0, 0.94, 0.82) * sun_disc * 9.0 + vec3<f32>(1.0, 0.78, 0.52) * sun_halo;

    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Curved Planetary Terrain & Orbital Sky-Ring Islands Pass
// -----------------------------------------------------------------------------
struct VertIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: vec3<f32>,
    @location(4) mat: vec4<f32>, // roughness, metallic, water_depth, ice_frac
};

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: vec3<f32>,
    @location(4) mat: vec4<f32>,
};

@vertex
fn vs_planet(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos = uni.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.normal = in.normal;
    out.color = in.color;
    out.emissive = in.emissive;
    out.mat = in.mat;
    return out;
}

@fragment
fn fs_terrain(in: VertOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(uni.cam_pos.xyz - p);

    // Crisp seamless 3D spherical micro-texture detail (zero UV seams!)
    let micro = fbm3(p * 0.68);
    let micro_fine = noise3(p * 2.8);
    let detail_mod = 0.78 + 0.34 * micro + 0.10 * (micro_fine - 0.5);

    var n = normalize(in.normal);
    // Rich procedural slope & crag bump perturbation
    let eps = 0.20;
    let gx = fbm3((p + vec3<f32>(eps, 0.0, 0.0)) * 0.95) - fbm3((p - vec3<f32>(eps, 0.0, 0.0)) * 0.95);
    let gy = fbm3((p + vec3<f32>(0.0, eps, 0.0)) * 0.95) - fbm3((p - vec3<f32>(0.0, eps, 0.0)) * 0.95);
    let gz = fbm3((p + vec3<f32>(0.0, 0.0, eps)) * 0.95) - fbm3((p - vec3<f32>(0.0, 0.0, eps)) * 0.95);
    let grad = vec3<f32>(gx, gy, gz);
    n = normalize(n - (grad - radial * dot(grad, radial)) * 0.95);

    // Expose crisp granite rock faces on steep mountain slopes
    let slope = clamp(1.0 - max(dot(n, radial), 0.0), 0.0, 1.0);
    let rock_col = vec3<f32>(0.30, 0.31, 0.35) * (0.85 + 0.30 * micro_fine);
    let is_emissive = step(0.15, length(in.emissive));
    let base_albedo = mix(in.color, rock_col, smoothstep(0.18, 0.45, slope) * (1.0 - is_emissive) * 0.72);

    // Day/Night Terminator illumination using planetary radial terminator + surface normal
    let term_raw = dot(radial, sun);
    let day_mask = smoothstep(-0.14, 0.22, term_raw);
    let twilight = exp(-pow((term_raw - 0.02) * 5.2, 2.0));

    let ndl = max(dot(n, sun), 0.0);
    let wrap_ndl = max((dot(n, sun) + 0.14) / 1.14, 0.0);

    // Sun color shifts warmly through golden-amber at the terminator!
    let sun_col = mix(vec3<f32>(1.0, 0.46, 0.16), vec3<f32>(1.06, 0.98, 0.88), smoothstep(-0.04, 0.30, term_raw));
    let diffuse = sun_col * (ndl * 1.55 * day_mask + wrap_ndl * 0.32 * twilight);

    // Sky hemisphere ambient (rich azure on day side, subtle starlight indigo on night side)
    let sky_up = 0.5 + 0.5 * dot(n, radial);
    let day_amb = mix(vec3<f32>(0.03, 0.05, 0.09), vec3<f32>(0.11, 0.20, 0.36), sky_up) * (0.20 + 0.80 * day_mask);
    let night_amb = vec3<f32>(0.010, 0.016, 0.032) * (1.0 - day_mask);

    // Specular highlight (stronger on glacier ice & wet alpine rock)
    let half_v = normalize(sun + view_dir);
    let rough = clamp(in.mat.x, 0.12, 0.96);
    let shininess = mix(96.0, 6.0, rough);
    let spec = pow(max(dot(n, half_v), 0.0), shininess) * mix(0.42, 0.04, rough) * day_mask;

    // Subtle atmospheric bounce rim on sunlit planetary edges
    let fres = pow(1.0 - max(dot(n, view_dir), 0.0), 3.4);
    let rim_col = mix(vec3<f32>(0.10, 0.32, 0.72), vec3<f32>(0.95, 0.42, 0.15), twilight) * fres * (0.18 * day_mask + 0.24 * twilight);

    let albedo = base_albedo * detail_mod;
    // Emissive lava/crystal glow shines brilliantly on night side and pulses subtly with micro noise
    let em_pulse = 0.85 + 0.35 * smoothstep(0.25, 0.85, micro);
    let emissive = in.emissive * em_pulse * mix(1.65, 1.05, day_mask);

    let lit = albedo * (diffuse + day_amb + night_amb) + sun_col * spec + rim_col + emissive;
    return vec4<f32>(lit, 1.0);
}

// -----------------------------------------------------------------------------
// Curved Specular/Fresnel Sapphire Ocean & Frozen Sea-Ice Pass
// -----------------------------------------------------------------------------
@fragment
fn fs_ocean(in: VertOut) -> @location(0) vec4<f32> {
    let water_depth = in.mat.z;
    if (water_depth < 0.08) {
        discard;
    }
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(uni.cam_pos.xyz - p);

    let ice_frac = clamp(in.mat.w, 0.0, 1.0);
    let t = uni.cam_pos.w;

    // Multi-octave spherical ocean micro-waves
    let w1 = fbm3(p * 1.15 + vec3<f32>(t * 0.25, 0.0, -t * 0.18));
    let w2 = fbm3(p * 2.65 + vec3<f32>(-t * 0.20, t * 0.15, t * 0.22));
    let eps = 0.18;
    let gx = fbm3((p + vec3<f32>(eps, 0.0, 0.0)) * 1.65) - fbm3((p - vec3<f32>(eps, 0.0, 0.0)) * 1.65);
    let gy = fbm3((p + vec3<f32>(0.0, eps, 0.0)) * 1.65) - fbm3((p - vec3<f32>(0.0, eps, 0.0)) * 1.65);
    let gz = fbm3((p + vec3<f32>(0.0, 0.0, eps)) * 1.65) - fbm3((p - vec3<f32>(0.0, 0.0, eps)) * 1.65);
    let wave_g = vec3<f32>(gx, gy, gz);
    let wave_t = wave_g - radial * dot(wave_g, radial);
    let n = normalize(radial - wave_t * mix(0.55, 0.16, ice_frac));

    let term_raw = dot(radial, sun);
    let day_mask = smoothstep(-0.14, 0.22, term_raw);
    let twilight = exp(-pow((term_raw - 0.02) * 5.2, 2.0));
    let sun_col = mix(vec3<f32>(1.0, 0.46, 0.16), vec3<f32>(1.05, 0.98, 0.90), smoothstep(-0.04, 0.30, term_raw));

    // Depth-based ocean color: luminous turquoise shallows -> deep royal sapphire -> abyssal navy
    let shallow_col = vec3<f32>(0.03, 0.44, 0.62);
    let mid_col = vec3<f32>(0.014, 0.14, 0.42);
    let deep_col = vec3<f32>(0.006, 0.042, 0.19);
    let d_norm = clamp(water_depth / 2.2, 0.0, 1.0);
    var water_col = mix(shallow_col, mid_col, smoothstep(0.0, 0.38, d_norm));
    water_col = mix(water_col, deep_col, smoothstep(0.32, 1.0, d_norm));

    // Crisp coastline foam where water meets land
    let foam = smoothstep(0.45, 0.08, water_depth) * smoothstep(0.32, 0.72, w1 * 0.6 + w2 * 0.4);
    water_col = mix(water_col, vec3<f32>(0.84, 0.94, 0.99), foam * 0.65);

    // Polar & night-side sea-ice overlay driven by live `SP_ICE` mass!
    let ice_col = vec3<f32>(0.78, 0.91, 0.99) * (0.90 + 0.14 * w1);
    let base_col = mix(water_col, ice_col, ice_frac);

    // Fresnel grazing sky & sunset reflection
    let ndv = max(dot(n, view_dir), 0.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 4.4);
    let sky_refl = mix(vec3<f32>(0.015, 0.03, 0.07), vec3<f32>(0.18, 0.48, 0.95), day_mask)
        + vec3<f32>(1.0, 0.45, 0.16) * twilight * 0.68;

    // Shimmering sun glint across curved ocean
    let half_v = normalize(sun + view_dir);
    let spec_sharp = pow(max(dot(n, half_v), 0.0), mix(160.0, 28.0, ice_frac)) * mix(2.2, 0.4, ice_frac);
    let spec_broad = pow(max(dot(n, half_v), 0.0), 22.0) * mix(0.28, 0.08, ice_frac);
    let spec = sun_col * (spec_sharp + spec_broad) * (day_mask + twilight * 0.45);

    let ndl = max(dot(n, sun), 0.0);
    let diff = base_col * (sun_col * ndl * 1.25 * day_mask + vec3<f32>(0.07, 0.14, 0.28) * (0.20 + 0.80 * day_mask) + vec3<f32>(0.008, 0.014, 0.032) * (1.0 - day_mask));

    let col = mix(diff, sky_refl, fresnel * mix(0.66, 0.22, ice_frac)) + spec;
    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Pass 2: Sleek Depth-Aware Physical Rayleigh + Mie Atmosphere & ACES Tonemap
// -----------------------------------------------------------------------------
@group(1) @binding(0) var hdr_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;

fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let h = b * b - c;
    if (h < 0.0) {
        return vec2<f32>(1e9, -1e9);
    }
    let s = sqrt(h);
    return vec2<f32>(-b - s, -b + s);
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_atmosphere_resolve(in: FullScreenOut) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.pos.xy);
    var hdr = textureLoad(hdr_tex, pix, 0).rgb;
    let depth = textureLoad(depth_tex, pix, 0);

    let ro = uni.cam_pos.xyz;
    let sun = normalize(uni.sun_dir.xyz);
    let r_surf = uni.planet_cfg.y; // ~63.2m
    let r_atmo = uni.planet_cfg.z; // ~74.5m sleek atmosphere shell!
    let exposure = uni.planet_cfg.w;

    let rd = ray_dir_from_ndc(in.ndc);
    var t_scene = 1e6;
    if (depth < 0.99995) {
        let p_hit = uni.inv_view_proj * vec4<f32>(in.ndc, depth, 1.0);
        let wh = p_hit.xyz / p_hit.w;
        t_scene = length(wh - ro);
    }

    let hit_atmo = intersect_sphere(ro, rd, r_atmo);
    if (hit_atmo.x < hit_atmo.y && hit_atmo.y > 0.0) {
        let t0 = max(hit_atmo.x, 0.0);
        let t1 = min(hit_atmo.y, t_scene);
        if (t1 > t0) {
            let steps = 14;
            let dt = (t1 - t0) / f32(steps);
            let h_scale_r = (r_atmo - r_surf) * 0.22;
            let h_scale_m = (r_atmo - r_surf) * 0.10;

            let beta_r = vec3<f32>(0.0048, 0.0145, 0.042);
            let beta_m = vec3<f32>(0.0042, 0.0036, 0.0030);

            let mu = dot(rd, sun);
            let phase_r = 0.75 * (1.0 + mu * mu);
            let g = 0.76;
            let phase_m = 1.5 * ((1.0 - g * g) / (2.0 + g * g)) * (1.0 + mu * mu)
                / pow(max(1.0 + g * g - 2.0 * g * mu, 0.06), 1.5);

            var sum_r = vec3<f32>(0.0);
            var sum_m = vec3<f32>(0.0);
            var od_r = 0.0;
            var od_m = 0.0;

            for (var i = 0; i < steps; i++) {
                let ts = t0 + (f32(i) + 0.5) * dt;
                let sp = ro + rd * ts;
                let r_cur = length(sp);
                let alt = max(r_cur - r_surf, 0.0);

                let d_r = exp(-alt / h_scale_r) * dt;
                let d_m = exp(-alt / h_scale_m) * dt;
                od_r += d_r;
                od_m += d_m;

                let s_cos = dot(sp / max(r_cur, 1.0), sun);
                let sun_vis = smoothstep(-0.20, 0.16, s_cos);
                let sunset_od = mix(2.4, 0.16, smoothstep(-0.10, 0.42, s_cos));
                let tau_sun = beta_r * (sunset_od * 14.0) + beta_m * (sunset_od * 6.0);
                let tau_cam = beta_r * od_r + beta_m * od_m;
                let atten = exp(-(tau_cam + tau_sun)) * sun_vis;

                sum_r += d_r * atten;
                sum_m += d_m * atten;
            }

            let is_sky = select(0.0, 1.0, depth >= 0.99995);
            // Crisp vivid limb halo on space rays; gentle aerial perspective on surface rays so terrain stays crystal-clear!
            let scatter_gain = mix(0.68, 1.75, is_sky);
            let inscatter = (sum_r * beta_r * phase_r * 1.4 + sum_m * beta_m * phase_m * 0.45) * scatter_gain;
            let trans = exp(-(beta_r * od_r * 0.30 + beta_m * od_m * 0.25));

            // Warm golden-amber twilight sunset glow right along the terminator limb
            let p_mid = ro + rd * (0.5 * (t0 + t1));
            let mid_r = length(p_mid);
            let limb_f = exp(-max(mid_r - r_surf, 0.0) / ((r_atmo - r_surf) * 0.24));
            let term_mid = dot(normalize(p_mid), sun);
            let dusk_glow = exp(-pow((term_mid - 0.0) * 5.0, 2.0)) * limb_f * vec3<f32>(1.0, 0.36, 0.09) * mix(0.14, 0.52, is_sky);

            hdr = hdr * mix(trans, vec3<f32>(1.0), is_sky * 0.85) + inscatter + dusk_glow;
        }
    }

    let mapped = aces_filmic(hdr * exposure);
    let srgb = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
"#;

// =============================================================================
// Seamless Curved Planetary Mesh Builders (Terrain + Orbital Sky-Ring + Ocean)
// =============================================================================

struct PlanetMeshes {
    terrain_verts: Vec<PlanetVertex>,
    terrain_indices: Vec<u32>,
    sky_ring_verts: Vec<PlanetVertex>,
    sky_ring_indices: Vec<u32>,
    ocean_verts: Vec<PlanetVertex>,
    ocean_indices: Vec<u32>,
}

#[derive(Clone, Copy)]
struct SurfaceSample {
    surf_z: f32,
    water_depth: f32,
    ice_frac: f32,
    color: [f32; 3],
    emissive: [f32; 3],
    roughness: f32,
}

fn sample_column(planet: &CubedSpherePlanet, face: usize, x: usize, y: usize, z_sea: usize, ring_z_floor: usize) -> SurfaceSample {
    let vf = &planet.faces[face];
    let idx = y * planet.face_size + x;

    let mut solid_z = 0usize;
    let mut found_solid = false;
    let mut water_top_z = 0usize;
    let mut water_mass_top = 0u32;
    let mut ice_mass_sea = 0u32;
    let mut max_temp = 28_500i64;
    let mut has_steam = false;

    for z in (0..ring_z_floor).rev() {
        let l = &vf.layers[z];
        let sm = (l.mass[SP_ROCK][idx] as u32)
            + (l.mass[SP_SOIL][idx] as u32)
            + (l.mass[SP_ICE][idx] as u32)
            + (l.mass[SP_WOOD][idx] as u32)
            + (l.mass[SP_PLANT][idx] as u32);
        let wm = l.mass[SP_WATER][idx] as u32;
        if l.mass[SP_STEAM][idx] > 80 || l.mass[SP_SMOKE][idx] > 80 {
            has_steam = true;
        }
        if wm >= 120 && water_top_z == 0 {
            water_top_z = z;
            water_mass_top = wm;
        }
        if z == z_sea || z + 1 == z_sea {
            ice_mass_sea += l.mass[SP_ICE][idx] as u32;
        }
        if sm >= 180 && !found_solid {
            solid_z = z;
            found_solid = true;
            max_temp = l.cell_temp_ck(idx);
            break;
        }
    }

    let l = &vf.layers[solid_z];
    let rock = l.mass[SP_ROCK][idx] as f32;
    let soil = l.mass[SP_SOIL][idx] as f32;
    let plant = l.mass[SP_PLANT][idx] as f32;
    let wood = l.mass[SP_WOOD][idx] as f32;
    let ice = l.mass[SP_ICE][idx] as f32;
    let oil = l.mass[SP_OIL][idx] as f32;

    // Also inspect layer right above for glowing magma/oil surface
    let mut temp_ck = max_temp;
    if solid_z + 1 < ring_z_floor {
        let lu = &vf.layers[solid_z + 1];
        if lu.mass[SP_OIL][idx] > 100 || lu.cell_temp_ck(idx) > temp_ck {
            temp_ck = temp_ck.max(lu.cell_temp_ck(idx));
        }
    }

    let z_sea_f = z_sea as f32 + 0.55;
    let surf_z = if found_solid {
        // Sub-voxel height modulation by total solid fill for smooth coastal slopes
        let fill = ((rock + soil + ice + plant * 0.4 + wood * 0.35) / 1150.0).clamp(0.15, 1.0);
        solid_z as f32 + fill
    } else {
        10.0
    };

    let water_depth = (z_sea_f - surf_z).max(0.0);
    let ice_frac = if water_depth > 0.05 {
        let tot = (water_mass_top + ice_mass_sea).max(1) as f32;
        (ice_mass_sea as f32 / tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Determine rich physically-plausible surface albedo, roughness, and emissive glow!
    let mut color = [0.22f32, 0.25, 0.28];
    let mut emissive = [0.0f32; 3];
    let mut roughness = 0.78f32;

    if temp_ck >= 70_000 || (oil > 180.0 && temp_ck >= 52_000) {
        // Blazing incandescent equatorial volcanic magma/lava rift channel!
        let hot = ((temp_ck - 55_000) as f32 / 65_000.0).clamp(0.15, 1.0);
        color = [0.16, 0.05, 0.03];
        emissive = [
            3.8 * hot + 0.9,
            1.25 * hot * hot + 0.18 * hot,
            0.16 * hot * hot * hot,
        ];
        if has_steam {
            emissive[0] += 0.45;
            emissive[1] += 0.18;
        }
        roughness = 0.55;
    } else if temp_ck >= 45_000 && rock > 350.0 {
        // Scorched obsidian caldera rim with faint ember cracks
        let ember = ((temp_ck - 45_000) as f32 / 25_000.0).clamp(0.0, 1.0);
        color = [0.10, 0.09, 0.11];
        emissive = [0.65 * ember, 0.16 * ember, 0.03 * ember];
        roughness = 0.62;
    } else if ice >= 350.0 {
        // Gleaming polar glacier ice cap or high alpine snow peak
        let snow_mix = (ice / (ice + rock + 1.0)).clamp(0.0, 1.0);
        color = [
            0.66 + 0.28 * snow_mix,
            0.78 + 0.19 * snow_mix,
            0.88 + 0.11 * snow_mix,
        ];
        roughness = 0.24;
    } else if surf_z <= z_sea_f {
        // Submerged sandy/rocky seafloor
        color = [0.18, 0.26, 0.28];
        roughness = 0.85;
    } else if wood >= 240.0 {
        // Deep Nordic alpine pine forest canopy
        color = [0.055, 0.20, 0.095];
        roughness = 0.86;
    } else if plant >= 180.0 {
        // Rich emerald temperate continent meadow + golden coastal rim
        let alt_t = ((surf_z - z_sea_f) / 4.2).clamp(0.0, 1.0);
        if surf_z < z_sea_f + 0.55 {
            color = [0.56, 0.49, 0.34]; // Warm sandy coastline
        } else {
            color = [
                0.08 + 0.16 * alt_t,
                0.31 - 0.07 * alt_t,
                0.11 + 0.04 * alt_t,
            ];
        }
        roughness = 0.82;
    } else if soil >= 200.0 {
        // Warm golden coastal sand / highland soil
        color = [0.54, 0.47, 0.34];
        roughness = 0.88;
    } else if rock >= 200.0 {
        // Exposed granite alpine cliff
        color = [0.38, 0.40, 0.44];
        roughness = 0.72;
    }

    SurfaceSample {
        surf_z,
        water_depth,
        ice_frac,
        color,
        emissive,
        roughness,
    }
}

/// Bilinearly sample the live `CubedSpherePlanet` surface purely as a function of 3D unit
/// direction `dir` (crossing cube face boundaries seamlessly via `sphere_dir_to_face_uv`!).
fn sample_planet_surface_smooth(planet: &CubedSpherePlanet, dir: [f32; 3], z_sea: usize, ring_z_floor: usize) -> SurfaceSample {
    let n = planet.face_size;
    let nf = n as f32;
    let (face, u, v) = sphere_dir_to_face_uv(dir);
    let gx = ((u + 1.0) * 0.5 * nf) - 0.5;
    let gy = ((v + 1.0) * 0.5 * nf) - 0.5;
    let ix0 = gx.floor() as i32;
    let iy0 = gy.floor() as i32;
    let fx = (gx - ix0 as f32).clamp(0.0, 1.0);
    let fy = (gy - iy0 as f32).clamp(0.0, 1.0);

    let fetch = |cx: i32, cy: i32| -> SurfaceSample {
        if cx >= 0 && cx < n as i32 && cy >= 0 && cy < n as i32 {
            sample_column(planet, face, cx as usize, cy as usize, z_sea, ring_z_floor)
        } else {
            // Seam-safe lookup across adjacent cube face via 3D projection!
            let tu = ((cx as f32 + 0.5) / nf) * 2.0 - 1.0;
            let tv = ((cy as f32 + 0.5) / nf) * 2.0 - 1.0;
            let d_nb = face_uv_to_sphere_dir(face, tu, tv);
            let (f2, x2, y2) = planet.sample_dir_cell(d_nb);
            sample_column(planet, f2, x2, y2, z_sea, ring_z_floor)
        }
    };

    let s00 = fetch(ix0, iy0);
    let s10 = fetch(ix0 + 1, iy0);
    let s01 = fetch(ix0, iy0 + 1);
    let s11 = fetch(ix0 + 1, iy0 + 1);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let blend = |a: f32, b: f32, c: f32, d: f32| a * w00 + b * w10 + c * w01 + d * w11;
    let mut color = [0.0f32; 3];
    let mut emissive = [0.0f32; 3];
    for ch in 0..3 {
        color[ch] = blend(s00.color[ch], s10.color[ch], s01.color[ch], s11.color[ch]);
        emissive[ch] = blend(s00.emissive[ch], s10.emissive[ch], s01.emissive[ch], s11.emissive[ch]);
    }
    SurfaceSample {
        surf_z: blend(s00.surf_z, s10.surf_z, s01.surf_z, s11.surf_z),
        water_depth: blend(s00.water_depth, s10.water_depth, s01.water_depth, s11.water_depth),
        ice_frac: blend(s00.ice_frac, s10.ice_frac, s01.ice_frac, s11.ice_frac),
        color,
        emissive,
        roughness: blend(s00.roughness, s10.roughness, s01.roughness, s11.roughness),
    }
}

fn build_planet_meshes(planet: &CubedSpherePlanet) -> PlanetMeshes {
    let n = planet.face_size;
    let d = planet.depth;
    let r_core = planet.r_core;
    let z_sea = ((15.0 * (d as f32 / 32.0)).round() as usize).clamp(4, d - 4);
    let ring_z_floor = d.saturating_sub(10);

    // -------------------------------------------------------------------------
    // 1. Seamless Curved Planetary Terrain Shell (`6 × 128 × 128` smooth quads)
    // -------------------------------------------------------------------------
    let sub = (n * 2).min(128);
    let stride = sub + 1;
    let mut terrain_verts = Vec::with_capacity(FACE_COUNT * stride * stride);
    let mut terrain_indices = Vec::with_capacity(FACE_COUNT * sub * sub * 6);

    for f in 0..FACE_COUNT {
        let base_v = terrain_verts.len() as u32;
        for iy in 0..=sub {
            let v = (iy as f32 / sub as f32) * 2.0 - 1.0;
            for ix in 0..=sub {
                let u = (ix as f32 / sub as f32) * 2.0 - 1.0;
                let dir = face_uv_to_sphere_dir(f, u, v);
                let dir_v = Vec3::from_array(dir);
                let samp = sample_planet_surface_smooth(planet, dir, z_sea, ring_z_floor);
                let r = r_core + samp.surf_z;
                let pos = dir_v * r;

                // Seamless 3D analytical spherical tangent gradient normal (identical across all 12 edges!)
                let ref_up = if dir_v.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let t1 = ref_up.cross(dir_v).normalize();
                let t2 = dir_v.cross(t1).normalize();
                let eps = 0.014f32;
                let d_p1 = (dir_v + t1 * eps).normalize();
                let d_m1 = (dir_v - t1 * eps).normalize();
                let d_p2 = (dir_v + t2 * eps).normalize();
                let d_m2 = (dir_v - t2 * eps).normalize();

                let r_p1 = r_core + sample_planet_surface_smooth(planet, d_p1.to_array(), z_sea, ring_z_floor).surf_z;
                let r_m1 = r_core + sample_planet_surface_smooth(planet, d_m1.to_array(), z_sea, ring_z_floor).surf_z;
                let r_p2 = r_core + sample_planet_surface_smooth(planet, d_p2.to_array(), z_sea, ring_z_floor).surf_z;
                let r_m2 = r_core + sample_planet_surface_smooth(planet, d_m2.to_array(), z_sea, ring_z_floor).surf_z;

                let dp1 = d_p1 * r_p1 - d_m1 * r_m1;
                let dp2 = d_p2 * r_p2 - d_m2 * r_m2;
                let mut normal = dp1.cross(dp2).normalize_or_zero();
                if normal.dot(dir_v) < 0.0 {
                    normal = -normal;
                }
                if normal.length_squared() < 0.5 {
                    normal = dir_v;
                }

                terrain_verts.push(PlanetVertex {
                    pos: pos.to_array(),
                    normal: normal.to_array(),
                    color: samp.color,
                    emissive: samp.emissive,
                    roughness: samp.roughness,
                    metallic: 0.0,
                    water_depth: samp.water_depth,
                    ice_frac: samp.ice_frac,
                });
            }
        }
        // Emit outward CCW triangles (`e_u × e_v == n` on every face -> `(00, 10, 11)` and `(00, 11, 01)`!)
        for iy in 0..sub {
            for ix in 0..sub {
                let i00 = base_v + (iy * stride + ix) as u32;
                let i10 = i00 + 1;
                let i01 = base_v + ((iy + 1) * stride + ix) as u32;
                let i11 = i01 + 1;
                terrain_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Curved Specular/Fresnel Sapphire Ocean Shell (`6 × 64 × 64` curved quads)
    // -------------------------------------------------------------------------
    let o_sub = n.min(64);
    let o_stride = o_sub + 1;
    let r_ocean = r_core + z_sea as f32 + 0.55;
    let mut ocean_verts = Vec::with_capacity(FACE_COUNT * o_stride * o_stride);
    let mut ocean_indices = Vec::with_capacity(FACE_COUNT * o_sub * o_sub * 6);

    for f in 0..FACE_COUNT {
        let base_v = ocean_verts.len() as u32;
        for iy in 0..=o_sub {
            let v = (iy as f32 / o_sub as f32) * 2.0 - 1.0;
            for ix in 0..=o_sub {
                let u = (ix as f32 / o_sub as f32) * 2.0 - 1.0;
                let dir = face_uv_to_sphere_dir(f, u, v);
                let samp = sample_planet_surface_smooth(planet, dir, z_sea, ring_z_floor);
                ocean_verts.push(PlanetVertex {
                    pos: [dir[0] * r_ocean, dir[1] * r_ocean, dir[2] * r_ocean],
                    normal: dir,
                    color: [0.04, 0.24, 0.54],
                    emissive: [0.0; 3],
                    roughness: 0.08,
                    metallic: 0.1,
                    water_depth: samp.water_depth,
                    ice_frac: samp.ice_frac,
                });
            }
        }
        for iy in 0..o_sub {
            for ix in 0..o_sub {
                let i00 = base_v + (iy * o_stride + ix) as u32;
                let i10 = i00 + 1;
                let i01 = base_v + ((iy + 1) * o_stride + ix) as u32;
                let i11 = i01 + 1;
                ocean_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Sculpted Curved Orbital Floating Sky-Ring Islands (Smooth Dual-Shell Meshes
    //    driven directly by live `CubedSpherePlanet` `BuoyantCore`s & ring voxels!)
    //    - Zero blocky Minecraft stairs, zero edge-on gaps!
    //    - Lush emerald/alpine sculpted upper plateau + tapered rocky basalt underside
    //      with a radiant glowing cyan-amethyst Buoyant Crystal Core spire!
    // -------------------------------------------------------------------------
    let mut sky_ring_verts: Vec<PlanetVertex> = Vec::new();
    let mut sky_ring_indices: Vec<u32> = Vec::new();
    let nf = n as f32;

    for f in 0..FACE_COUNT {
        let vf = &planet.faces[f];
        for (ci, core) in vf.structural.buoyant_cores.iter().enumerate() {
            if core.current_core_mass <= 0 {
                continue;
            }
            let r_base = core.island_radius * 0.92;
            let z_top_base = core.cz + 4.4;
            let z_bot_tip = core.cz - 2.6;
            let phase = (f as f32) * 1.7 + (ci as f32) * 2.4;

            let rings = 10usize;
            let segs = 28usize;

            let island_radius_at = |ang: f32| -> f32 {
                r_base
                    * (1.0
                        + 0.18 * (ang * 3.0 + phase).sin()
                        + 0.11 * (ang * 5.0 - phase * 1.3).cos()
                        + 0.06 * (ang * 7.0 + phase * 0.7).sin())
            };

            let eval_top = |t: f32, ang: f32| -> (Vec3, [f32; 3], [f32; 3], f32) {
                let rad = island_radius_at(ang) * t;
                let gx = core.cx + rad * ang.cos();
                let gy = core.cy + rad * ang.sin();
                // Sculpted gentle central highland hill on top of the floating island
                let hill = (1.0 - t * t).powf(1.2) * 1.85
                    + 0.35 * (ang * 4.0 + phase).sin() * t * (1.0 - t);
                let z_f = z_top_base + hill + 1.8;
                let u = (gx / nf) * 2.0 - 1.0;
                let v = (gy / nf) * 2.0 - 1.0;
                let pos = Vec3::from_array(face_uv_z_to_world_pos(f, u, v, z_f, r_core));
                let col = if t > 0.86 {
                    [0.32, 0.33, 0.37] // Rocky cliff rim
                } else if t < 0.42 {
                    [0.07, 0.25, 0.11] // Deep emerald pine crown
                } else {
                    [0.14, 0.40, 0.17] // Lush floating meadow
                };
                (pos, col, [0.0; 3], 0.80)
            };

            let eval_bot = |t: f32, ang: f32| -> (Vec3, [f32; 3], [f32; 3], f32) {
                let rad = island_radius_at(ang) * t;
                let gx = core.cx + rad * ang.cos();
                let gy = core.cy + rad * ang.sin();
                // Tapered craggy underside keel descending to a glowing Buoyant Crystal Core point!
                let crag = 0.42 * (ang * 6.0 - phase).sin() * t * (1.0 - t);
                let z_f = z_bot_tip + (z_top_base - z_bot_tip) * t.powf(0.58) + crag + 1.8;
                let u = (gx / nf) * 2.0 - 1.0;
                let v = (gy / nf) * 2.0 - 1.0;
                let pos = Vec3::from_array(face_uv_z_to_world_pos(f, u, v, z_f, r_core));

                let core_glow = (1.0 - (t / 0.52).clamp(0.0, 1.0)).powf(1.5) * core.integrity;
                let col = [
                    0.20 + 0.25 * core_glow,
                    0.22 + 0.65 * core_glow,
                    0.30 + 0.70 * core_glow,
                ];
                let em = [
                    0.42 * core_glow + 0.04,
                    1.65 * core_glow + 0.10,
                    2.95 * core_glow + 0.18,
                ];
                (pos, col, em, 0.55 - 0.30 * core_glow)
            };

            // Emit upper curved shell (outward CCW) and lower tapered crystal keel shell (inward/downward CCW)
            for is_top in [true, false] {
                let base_idx = sky_ring_verts.len() as u32;
                let stride_s = segs + 1;
                for ri in 0..=rings {
                    let t = ri as f32 / rings as f32;
                    for si in 0..=segs {
                        let ang = (si as f32 / segs as f32) * std::f32::consts::TAU;
                        let (pos, col, em, rough) = if is_top {
                            eval_top(t, ang)
                        } else {
                            eval_bot(t, ang)
                        };
                        // Analytical finite-difference normal on the curved island surface!
                        let dt = 0.03f32;
                        let da = 0.06f32;
                        let t_a = (t + dt).min(1.0);
                        let t_b = (t - dt).max(0.0);
                        let p_ta = if is_top { eval_top(t_a, ang).0 } else { eval_bot(t_a, ang).0 };
                        let p_tb = if is_top { eval_top(t_b, ang).0 } else { eval_bot(t_b, ang).0 };
                        let p_aa = if is_top { eval_top(t.max(0.04), ang + da).0 } else { eval_bot(t.max(0.04), ang + da).0 };
                        let p_ab = if is_top { eval_top(t.max(0.04), ang - da).0 } else { eval_bot(t.max(0.04), ang - da).0 };
                        let rad_dir = pos.normalize();
                        let mut norm = (p_ta - p_tb).cross(p_aa - p_ab).normalize_or_zero();
                        let want_sign = if is_top { 1.0 } else { -1.0 };
                        if norm.length_squared() < 0.25 {
                            norm = rad_dir * want_sign;
                        } else if norm.dot(rad_dir) * want_sign < 0.0 {
                            norm = -norm;
                        }
                        sky_ring_verts.push(PlanetVertex {
                            pos: pos.to_array(),
                            normal: norm.to_array(),
                            color: col,
                            emissive: em,
                            roughness: rough,
                            metallic: 0.0,
                            water_depth: 0.0,
                            ice_frac: 0.0,
                        });
                    }
                }
                for ri in 0..rings {
                    for si in 0..segs {
                        let i00 = base_idx + (ri * stride_s + si) as u32;
                        let i10 = base_idx + ((ri + 1) * stride_s + si) as u32;
                        let i01 = i00 + 1;
                        let i11 = i10 + 1;
                        if is_top {
                            sky_ring_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                        } else {
                            sky_ring_indices.extend_from_slice(&[i00, i11, i10, i00, i01, i11]);
                        }
                    }
                }
            }
        }
    }

    PlanetMeshes {
        terrain_verts,
        terrain_indices,
        sky_ring_verts,
        sky_ring_indices,
        ocean_verts,
        ocean_indices,
    }
}

// =============================================================================
// Headless Apple M4 Pro Metal Planetary Renderer
// =============================================================================

struct MetalPlanetRenderer {
    adapter_name: String,
    adapter_backend: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    uni_buf: wgpu::Buffer,
    bg_uni: wgpu::BindGroup,
    resolve_bgl: wgpu::BindGroupLayout,

    pipe_starfield: wgpu::RenderPipeline,
    pipe_terrain: wgpu::RenderPipeline,
    pipe_ocean: wgpu::RenderPipeline,
    pipe_resolve: wgpu::RenderPipeline,

    terrain_vbuf: wgpu::Buffer,
    terrain_ibuf: wgpu::Buffer,
    terrain_icount: u32,

    sky_vbuf: wgpu::Buffer,
    sky_ibuf: wgpu::Buffer,
    sky_icount: u32,

    ocean_vbuf: wgpu::Buffer,
    ocean_ibuf: wgpu::Buffer,
    ocean_icount: u32,

    r_core: f32,
    r_ocean: f32,
    r_atmo: f32,

    target_w: u32,
    target_h: u32,
    hdr_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    bg_resolve: wgpu::BindGroup,
}

fn make_hdr_depth(
    device: &wgpu::Device,
    resolve_bgl: &wgpu::BindGroupLayout,
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
    let size = wgpu::Extent3d {
        width: w.max(1),
        height: h.max(1),
        depth_or_array_layers: 1,
    };
    let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_hdr"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let hdr_view = hdr_tex.create_view(&Default::default());
    let depth_view = depth_tex.create_view(&Default::default());
    let bg_resolve = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("planet_resolve_bg"),
        layout: resolve_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            },
        ],
    });
    (hdr_view, depth_view, bg_resolve)
}

fn create_buf<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(data);
    let size = (bytes.len() as u64).max(16);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buf, 0, bytes);
    }
    buf
}

impl MetalPlanetRenderer {
    async fn new(meshes: &PlanetMeshes, r_core: f32, depth: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await?;
        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let adapter_backend = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("planet_metal_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PLANET_SHADER_WGSL)),
        });

        let uni_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet_uni_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet_resolve_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        let uni_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet_uni_buf"),
            size: std::mem::size_of::<PlanetUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_uni = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet_uni_bg"),
            layout: &uni_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uni_buf.as_entire_binding(),
            }],
        });

        let hdr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet_hdr_pl"),
            bind_group_layouts: &[Some(&uni_bgl)],
            immediate_size: 0,
        });

        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet_resolve_pl"),
            bind_group_layouts: &[Some(&uni_bgl), Some(&resolve_bgl)],
            immediate_size: 0,
        });

        let pipe_starfield = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_starfield"),
            layout: Some(&hdr_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_starfield"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_terrain = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_terrain"),
            layout: Some(&hdr_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_planet"),
                buffers: &[Some(PlanetVertex::desc())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_terrain"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_ocean = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_ocean"),
            layout: Some(&hdr_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_planet"),
                buffers: &[Some(PlanetVertex::desc())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ocean"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipe_resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_resolve"),
            layout: Some(&resolve_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_atmosphere_resolve"),
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

        let terrain_vbuf = create_buf(&device, &queue, "terrain_vbuf", &meshes.terrain_verts, wgpu::BufferUsages::VERTEX);
        let terrain_ibuf = create_buf(&device, &queue, "terrain_ibuf", &meshes.terrain_indices, wgpu::BufferUsages::INDEX);
        let sky_vbuf = create_buf(&device, &queue, "sky_vbuf", &meshes.sky_ring_verts, wgpu::BufferUsages::VERTEX);
        let sky_ibuf = create_buf(&device, &queue, "sky_ibuf", &meshes.sky_ring_indices, wgpu::BufferUsages::INDEX);
        let ocean_vbuf = create_buf(&device, &queue, "ocean_vbuf", &meshes.ocean_verts, wgpu::BufferUsages::VERTEX);
        let ocean_ibuf = create_buf(&device, &queue, "ocean_ibuf", &meshes.ocean_indices, wgpu::BufferUsages::INDEX);

        let (hdr_view, depth_view, bg_resolve) = make_hdr_depth(&device, &resolve_bgl, 1920, 1080);

        let z_sea = ((15.0 * (depth as f32 / 32.0)).round() as usize).clamp(4, depth - 4);
        let r_ocean = r_core + z_sea as f32 + 0.55;
        // Sleek atmospheric shell hugging the curved planet (`~10.5m` above sea level)
        let r_atmo = r_ocean + 10.5;

        Ok(Self {
            adapter_name,
            adapter_backend,
            device,
            queue,
            uni_buf,
            bg_uni,
            resolve_bgl,
            pipe_starfield,
            pipe_terrain,
            pipe_ocean,
            pipe_resolve,
            terrain_vbuf,
            terrain_ibuf,
            terrain_icount: meshes.terrain_indices.len() as u32,
            sky_vbuf,
            sky_ibuf,
            sky_icount: meshes.sky_ring_indices.len() as u32,
            ocean_vbuf,
            ocean_ibuf,
            ocean_icount: meshes.ocean_indices.len() as u32,
            r_core,
            r_ocean,
            r_atmo,
            target_w: 1920,
            target_h: 1080,
            hdr_view,
            depth_view,
            bg_resolve,
        })
    }

    fn ensure_targets(&mut self, w: u32, h: u32) {
        if self.target_w != w || self.target_h != h {
            let (hv, dv, bg) = make_hdr_depth(&self.device, &self.resolve_bgl, w, h);
            self.hdr_view = hv;
            self.depth_view = dv;
            self.bg_resolve = bg;
            self.target_w = w;
            self.target_h = h;
        }
    }

    fn render_into(&mut self, out_view: &wgpu::TextureView, w: u32, h: u32, shot: &CameraShot, time_s: f32) {
        self.ensure_targets(w, h);
        let (vp, inv_vp) = shot.view_proj(w, h);
        let uni = PlanetUniforms {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: inv_vp.to_cols_array_2d(),
            cam_pos: [shot.pos.x, shot.pos.y, shot.pos.z, time_s],
            sun_dir: [shot.sun_dir.x, shot.sun_dir.y, shot.sun_dir.z, self.r_ocean],
            planet_cfg: [self.r_core, self.r_ocean - 0.35, self.r_atmo, shot.exposure],
            viewport: [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
        };
        self.queue.write_buffer(&self.uni_buf, 0, bytemuck::bytes_of(&uni));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet_enc"),
        });

        // Pass 1: Cosmic Starfield + Curved Planetary Terrain + Orbital Sky-Ring + Fresnel Ocean -> HDR + Depth
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("planet_hdr_pass"),
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

            // 1a. Starfield backdrop
            rp.set_pipeline(&self.pipe_starfield);
            rp.set_bind_group(0, &self.bg_uni, &[]);
            rp.draw(0..3, 0..1);

            // 1b. Curved Planetary Terrain + Orbital Floating Sky-Ring Islands
            rp.set_pipeline(&self.pipe_terrain);
            rp.set_bind_group(0, &self.bg_uni, &[]);
            rp.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
            rp.set_index_buffer(self.terrain_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.terrain_icount, 0, 0..1);

            if self.sky_icount > 0 {
                rp.set_vertex_buffer(0, self.sky_vbuf.slice(..));
                rp.set_index_buffer(self.sky_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.sky_icount, 0, 0..1);
            }

            // 1c. Curved Specular/Fresnel Sapphire Ocean & Polar Sea-Ice Shell
            rp.set_pipeline(&self.pipe_ocean);
            rp.set_bind_group(0, &self.bg_uni, &[]);
            rp.set_vertex_buffer(0, self.ocean_vbuf.slice(..));
            rp.set_index_buffer(self.ocean_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.ocean_icount, 0, 0..1);
        }

        // Pass 2: Physical Depth-Aware Rayleigh+Mie Atmosphere + ACES Filmic Tonemap -> Rgba8Unorm
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("planet_atmo_resolve_pass"),
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
            rp.set_pipeline(&self.pipe_resolve);
            rp.set_bind_group(0, &self.bg_uni, &[]);
            rp.set_bind_group(1, &self.bg_resolve, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(Some(enc.finish()));
    }

    fn create_out_tex(&self, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planet_out_rgba"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    fn render_to_image(&mut self, w: u32, h: u32, shot: &CameraShot, time_s: f32) -> RgbaImage {
        let (tex, view) = self.create_out_tex(w, h);
        self.render_into(&view, w, h, shot, time_s);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let row_bytes = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet_readback"),
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
        let mut raw = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let s = (y * padded) as usize;
            raw.extend_from_slice(&mapped[s..s + row_bytes as usize]);
        }
        drop(mapped);
        staging.unmap();
        RgbaImage::from_raw(w, h, raw).expect("valid RGBA image")
    }

    fn benchmark(&mut self, w: u32, h: u32, shot: &CameraShot, warmup: usize, iters: usize) -> ResBench {
        let (_tex, view) = self.create_out_tex(w, h);
        for i in 0..warmup {
            self.render_into(&view, w, h, shot, i as f32 * 0.016);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let mut samples = Vec::with_capacity(iters);
        for i in 0..iters {
            let t0 = Instant::now();
            self.render_into(&view, w, h, shot, (warmup + i) as f32 * 0.016);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = samples[samples.len() / 2];
        let p99 = samples[((samples.len() as f64) * 0.99)
            .floor()
            .min((samples.len() - 1) as f64) as usize];
        let mean = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
        ResBench {
            width: w,
            height: h,
            iters,
            p50_ms: round3(p50),
            p99_ms: round3(p99),
            mean_ms: round3(mean),
            fps_p50: round2(1000.0 / p50.max(1e-4)),
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

// =============================================================================
// Crisp 5×7 Bitmap Badges for 2×2 Showcase Composite
// =============================================================================

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
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
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
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b00100, 0b01000, 0b10000, 0b00000],
        '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
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
            if px < x0 + 4 {
                *p = Rgba([accent[0], accent[1], accent[2], 255]);
            } else {
                p[0] = ((p[0] as u32 * 56 + 10 * 199) / 255) as u8;
                p[1] = ((p[1] as u32 * 56 + 14 * 199) / 255) as u8;
                p[2] = ((p[2] as u32 * 56 + 24 * 199) / 255) as u8;
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
                if (bits >> (4 - rx)) & 1 != 0 {
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

fn blit_quadrant(dst: &mut RgbaImage, src: &RgbaImage, ox: u32, oy: u32) {
    let (sw, sh) = src.dimensions();
    for y in 0..sh {
        for x in 0..sw {
            dst.put_pixel(ox + x, oy + y, *src.get_pixel(x, y));
        }
    }
}

// =============================================================================
// Metrics JSON Schema
// =============================================================================

#[derive(Serialize)]
struct ResBench {
    width: u32,
    height: u32,
    iters: usize,
    p50_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    fps_p50: f64,
}

#[derive(Serialize)]
struct MeshStats {
    terrain_vertices: usize,
    terrain_triangles: usize,
    sky_ring_vertices: usize,
    sky_ring_triangles: usize,
    ocean_vertices: usize,
    ocean_triangles: usize,
    total_vertices: usize,
    total_triangles: usize,
}

#[derive(Serialize)]
struct PlanetMetalMetrics {
    wave: &'static str,
    adapter: String,
    backend: String,
    face_size: usize,
    depth: usize,
    r_core: f32,
    outer_radius: f32,
    active_voxels_6face: usize,
    sim_ticks: usize,
    planet_step_p50_ms: f64,
    planet_step_mean_ms: f64,
    global_mass_error: i64,
    global_energy_error: i64,
    total_seam_mass_crossed: i64,
    buoyant_core_count: usize,
    mesh_stats: MeshStats,
    bench_1080p: ResBench,
    bench_4k: ResBench,
    showcase_outputs: Vec<String>,
}

// =============================================================================
// Scenic Camera Picker for Low-Altitude Curved Surface Horizon Vista
// =============================================================================

fn build_horizon_camera(planet: &CubedSpherePlanet, sun_dir: Vec3) -> CameraShot {
    let d = planet.depth;
    let z_sea = ((15.0 * (d as f32 / 32.0)).round() as usize).clamp(4, d - 4);
    let ring_z_floor = d.saturating_sub(10);

    // Collect all Sky-Ring island directions so we stand at an ideal scenic distance (`0.48..0.82 rad`),
    // NEVER directly underneath an island blocking the view, looking across curved ocean & emerald peaks
    // toward the visibly curving horizon with 1-2 Orbital Sky-Ring Islands hovering majestically in the sky!
    let mut island_dirs: Vec<Vec3> = Vec::new();
    for &f in &[FACE_POS_Z, FACE_NEG_X, FACE_POS_X, FACE_NEG_Z] {
        for core in &planet.faces[f].structural.buoyant_cores {
            let cu = (core.cx / planet.face_size as f32) * 2.0 - 1.0;
            let cv = (core.cy / planet.face_size as f32) * 2.0 - 1.0;
            island_dirs.push(Vec3::from_array(face_uv_to_sphere_dir(f, cu, cv)));
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_site = Vec3::new(-0.35, 0.38, 0.85).normalize();
    let mut best_fwd = Vec3::new(0.88, -0.35, 0.32).normalize();

    for &island_dir in &island_dirs {
        if island_dir.dot(sun_dir) < 0.05 {
            continue;
        }
        let tang_east = Vec3::Y.cross(island_dir).normalize_or_zero();
        // Stand ~0.52..0.68 rad north/south-west of the island so it sits gracefully in the upper sky!
        for &(a_east, a_north) in &[
            (-0.46f32, 0.34f32),
            (0.46, 0.34),
            (-0.52, 0.26),
            (0.52, 0.26),
            (-0.42, -0.34),
            (0.42, -0.34),
        ] {
            let site_dir = (island_dir + tang_east * a_east + Vec3::Y * a_north).normalize();
            let sun_dot = site_dir.dot(sun_dir);
            if sun_dot < 0.30 {
                continue;
            }
            // Ensure NO other Sky-Island is directly overhead (< 0.36 rad from camera site!)
            let too_close_to_island = island_dirs.iter().any(|&id| id.dot(site_dir) > 0.935);
            if too_close_to_island {
                continue;
            }
            let s_here = sample_planet_surface_smooth(planet, site_dir.to_array(), z_sea, ring_z_floor);
            let fg_dir = (site_dir * 0.82 + island_dir * 0.18).normalize();
            let mid_dir = (site_dir * 0.52 + island_dir * 0.48).normalize();
            let s_fg = sample_planet_surface_smooth(planet, fg_dir.to_array(), z_sea, ring_z_floor);
            let s_mid = sample_planet_surface_smooth(planet, mid_dir.to_array(), z_sea, ring_z_floor);

            // Prefer standing on scenic sunlit emerald/alpine coast overlooking sparkling curved sapphire ocean!
            let fg_land = if s_fg.surf_z > z_sea as f32 + 0.8 { 2.8 } else { 0.4 };
            let mid_ocean = if s_mid.water_depth > 0.5 { 3.5 } else { 0.5 };
            let here_ok = if s_here.surf_z > z_sea as f32 + 0.5 { 2.0 } else { 0.5 };
            let score = fg_land + mid_ocean + here_ok + sun_dot * 2.2;
            if score > best_score {
                best_score = score;
                best_site = site_dir;
                best_fwd = (island_dir - site_dir * island_dir.dot(site_dir)).normalize();
            }
        }
    }

    let s_site = sample_planet_surface_smooth(planet, best_site.to_array(), z_sea, ring_z_floor);
    let r_cam = planet.r_core + s_site.surf_z.max(z_sea as f32 + 1.5) + 5.8;
    let cam_pos = best_site * r_cam;
    // Tilt gently downward (`-0.22 * best_site`) so foreground terrain + curved sapphire ocean occupy
    // the lower half, the visibly curving horizon & Rayleigh blue limb sweep across center,
    // and the sculpted Floating Orbital Sky-Ring Islands hover in the starry upper sky!
    let look_dir = (best_fwd - best_site * 0.22).normalize();
    let target = cam_pos + look_dir * 45.0;

    CameraShot {
        pos: cam_pos,
        target,
        up: best_site,
        fov_y_deg: 58.0,
        near: 0.4,
        far: 550.0,
        sun_dir,
        exposure: 1.08,
    }
}

fn find_repo_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("models").exists() && cwd.join("crates").exists() {
        return cwd;
    }
    if let Some(p1) = cwd.parent() {
        if let Some(p2) = p1.parent() {
            if p2.join("models").exists() && p2.join("crates").exists() {
                return p2.to_path_buf();
            }
        }
    }
    cwd
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = find_repo_root();
    let showcase_dir = repo_root.join("models").join("planet_showcase");
    std::fs::create_dir_all(&showcase_dir)?;

    // 1. Instantiate live 6-Face Cubed-Sphere Planet (`64×64×32×6 = 786,432` active 3D voxels, outer R=80m)
    let face_size = 64usize;
    let depth = 32usize;
    let r_core = 48.0f32;
    let seed = 29029u64;

    println!(
        "[planet_render] Generating 6-Face CubedSpherePlanet ({face_size}x{face_size}x{depth}x6 = {} voxels, R_core={r_core}m, R_outer={}m)...",
        6 * face_size * face_size * depth,
        r_core + depth as f32
    );
    let t_gen = Instant::now();
    let mut planet = CubedSpherePlanet::new(face_size, depth, r_core, seed);
    let hero_sun = Vec3::new(-0.78, 0.28, 0.56).normalize();
    planet.sun_dir = hero_sun.to_array();
    println!(
        "[planet_render] Planet generated in {:.1} ms | buoyant_cores={} | init mass_err={} energy_err={}",
        t_gen.elapsed().as_secs_f64() * 1000.0,
        planet.buoyant_core_count(),
        planet.global_mass_error(),
        planet.global_energy_error()
    );

    // 2. Run 10 full conservation-exact planetary physics + sweeping terminator ticks!
    let sim_ticks = 10usize;
    let mut step_times = Vec::with_capacity(sim_ticks);
    for t in 0..sim_ticks {
        let sweep = (t as f32 - 9.0) * 0.03;
        let s_vec = Vec3::new(
            hero_sun.x * sweep.cos() - hero_sun.z * sweep.sin(),
            hero_sun.y,
            hero_sun.x * sweep.sin() + hero_sun.z * sweep.cos(),
        )
        .normalize();
        planet.sun_dir = s_vec.to_array();
        let st = planet.step_planet();
        assert_eq!(st.global_mass_error, 0, "global_mass_error MUST be 0 at tick {}", st.tick);
        assert_eq!(st.global_energy_error, 0, "global_energy_error MUST be 0 at tick {}", st.tick);
        step_times.push(st.step_ms);
    }
    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let planet_step_p50_ms = step_times[step_times.len() / 2];
    let planet_step_mean_ms = step_times.iter().sum::<f64>() / step_times.len() as f64;
    println!(
        "[planet_render] 10 ticks complete: p50={planet_step_p50_ms:.2}ms mean={planet_step_mean_ms:.2}ms | seam_mass={} | mass_err={} energy_err={}",
        planet.total_seam_mass_crossed,
        planet.global_mass_error(),
        planet.global_energy_error()
    );

    // 3. Extract seamless curved 3D spherical meshes (terrain + ocean + Orbital Sky-Ring Islands)
    let t_mesh = Instant::now();
    let meshes = build_planet_meshes(&planet);
    let mesh_stats = MeshStats {
        terrain_vertices: meshes.terrain_verts.len(),
        terrain_triangles: meshes.terrain_indices.len() / 3,
        sky_ring_vertices: meshes.sky_ring_verts.len(),
        sky_ring_triangles: meshes.sky_ring_indices.len() / 3,
        ocean_vertices: meshes.ocean_verts.len(),
        ocean_triangles: meshes.ocean_indices.len() / 3,
        total_vertices: meshes.terrain_verts.len() + meshes.sky_ring_verts.len() + meshes.ocean_verts.len(),
        total_triangles: (meshes.terrain_indices.len() + meshes.sky_ring_indices.len() + meshes.ocean_indices.len()) / 3,
    };
    println!(
        "[planet_render] Curved 6-face mesh extracted in {:.1} ms: {} tris ({} terrain + {} sky-ring + {} ocean)",
        t_mesh.elapsed().as_secs_f64() * 1000.0,
        mesh_stats.total_triangles,
        mesh_stats.terrain_triangles,
        mesh_stats.sky_ring_triangles,
        mesh_stats.ocean_triangles
    );

    // 4. Initialize headless Apple M4 Pro Metal renderer
    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let mut renderer = rt.block_on(MetalPlanetRenderer::new(&meshes, r_core, depth))?;
    println!(
        "[planet_render] GPU: {} ({})",
        renderer.adapter_name, renderer.adapter_backend
    );

    // 5. Define the 4 iconic planetary camera shots
    // Shot 1: Hero Full-Globe Orbital View framing Day/Night Terminator, continents, Fresnel ocean,
    //         polar glacier cap, glowing night-side volcanic rift, Rayleigh limb & Orbital Sky-Ring!
    let shot_orbital = CameraShot {
        pos: Vec3::new(0.42, 0.24, 0.875).normalize() * 198.0,
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_deg: 44.0,
        near: 1.0,
        far: 900.0,
        sun_dir: hero_sun,
        exposure: 1.22,
    };

    // Shot 2: Dramatic Low-Altitude Curved Surface Horizon Vista looking over emerald terrain,
    //         curved sapphire ocean, bending horizon & overhead Orbital Floating Sky-Ring!
    let shot_horizon = build_horizon_camera(&planet, hero_sun);

    // Shot 3: Polar Glacier Cap + Equatorial Orbital Sky-Ring oblique orbital showcase
    let shot_polar = CameraShot {
        pos: Vec3::new(-0.34, 0.86, 0.38).normalize() * 194.0,
        target: Vec3::new(0.0, 4.0, 0.0),
        up: Vec3::new(0.0, 0.0, -1.0),
        fov_y_deg: 44.0,
        near: 1.0,
        far: 900.0,
        sun_dir: hero_sun,
        exposure: 1.18,
    };

    // Shot 4: Night-Side Volcanic Rift Glow + Thin Sunlit Rayleigh Crescent + Glowing Crystal Cores
    let shot_night_rift = CameraShot {
        pos: Vec3::new(0.86, 0.36, -0.36).normalize() * 190.0,
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_deg: 44.0,
        near: 1.0,
        far: 900.0,
        sun_dir: hero_sun,
        exposure: 1.28,
    };

    // 6. Render & save Hero 1080p PNGs + 2×2 4-View Showcase Composite
    let p_orbital = showcase_dir.join("planet_orbital_globe_1080p.png");
    let p_horizon = showcase_dir.join("planet_horizon_surface_1080p.png");
    let p_4view = showcase_dir.join("planet_4view_showcase.png");

    let img_orbital_1080 = renderer.render_to_image(1920, 1080, &shot_orbital, 1.0);
    img_orbital_1080.save(&p_orbital)?;
    println!("[planet_render] Saved {}", p_orbital.display());

    let img_horizon_1080 = renderer.render_to_image(1920, 1080, &shot_horizon, 1.0);
    img_horizon_1080.save(&p_horizon)?;
    println!("[planet_render] Saved {}", p_horizon.display());

    // Render 4 native 960×540 crisp quadrant frames for the 1920×1080 2×2 showcase
    let q_tl = renderer.render_to_image(960, 540, &shot_orbital, 1.0);
    let q_tr = renderer.render_to_image(960, 540, &shot_horizon, 1.0);
    let q_bl = renderer.render_to_image(960, 540, &shot_polar, 1.0);
    let q_br = renderer.render_to_image(960, 540, &shot_night_rift, 1.0);

    let mut composite = RgbaImage::new(1920, 1080);
    blit_quadrant(&mut composite, &q_tl, 0, 0);
    blit_quadrant(&mut composite, &q_tr, 960, 0);
    blit_quadrant(&mut composite, &q_bl, 0, 540);
    blit_quadrant(&mut composite, &q_br, 960, 540);

    // Subtle 2px divider lines between quadrants
    for x in 0..1920 {
        composite.put_pixel(x, 539, Rgba([28, 42, 68, 255]));
        composite.put_pixel(x, 540, Rgba([28, 42, 68, 255]));
    }
    for y in 0..1080 {
        composite.put_pixel(959, y, Rgba([28, 42, 68, 255]));
        composite.put_pixel(960, y, Rgba([28, 42, 68, 255]));
    }

    draw_badge(&mut composite, 16, 16, "DAY/NIGHT ORBITAL GLOBE", [76, 186, 255]);
    draw_badge(&mut composite, 976, 16, "CURVED SURFACE HORIZON", [72, 224, 152]);
    draw_badge(&mut composite, 16, 556, "POLAR CAP & EQUATORIAL RING", [165, 228, 255]);
    draw_badge(&mut composite, 976, 556, "NIGHT-SIDE VOLCANIC RIFT GLOW", [255, 134, 62]);
    composite.save(&p_4view)?;
    println!("[planet_render] Saved {}", p_4view.display());

    // 7. Benchmark 1080p & 4K Metal FPS (25 iters each)
    let bench_1080p = renderer.benchmark(1920, 1080, &shot_orbital, 5, 25);
    let bench_4k = renderer.benchmark(3840, 2160, &shot_orbital, 5, 25);
    println!(
        "[planet_render] 1080p Metal: p50={:.3}ms p99={:.3}ms ({:.1} FPS) | 4K Metal: p50={:.3}ms p99={:.3}ms ({:.1} FPS)",
        bench_1080p.p50_ms, bench_1080p.p99_ms, bench_1080p.fps_p50,
        bench_4k.p50_ms, bench_4k.p99_ms, bench_4k.fps_p50
    );

    // 8. Save `models/PLANET_METAL_METRICS.json`
    let metrics = PlanetMetalMetrics {
        wave: "Wave 29B: 6-Face Cubed-Sphere Exact Conservation & Curved Apple M4 Pro Metal Planetary Renderer",
        adapter: renderer.adapter_name,
        backend: renderer.adapter_backend,
        face_size,
        depth,
        r_core,
        outer_radius: r_core + depth as f32,
        active_voxels_6face: planet.total_voxels(),
        sim_ticks,
        planet_step_p50_ms: round3(planet_step_p50_ms),
        planet_step_mean_ms: round3(planet_step_mean_ms),
        global_mass_error: planet.global_mass_error(),
        global_energy_error: planet.global_energy_error(),
        total_seam_mass_crossed: planet.total_seam_mass_crossed,
        buoyant_core_count: planet.buoyant_core_count(),
        mesh_stats,
        bench_1080p,
        bench_4k,
        showcase_outputs: vec![
            "models/planet_showcase/planet_orbital_globe_1080p.png".to_string(),
            "models/planet_showcase/planet_horizon_surface_1080p.png".to_string(),
            "models/planet_showcase/planet_4view_showcase.png".to_string(),
        ],
    };

    let metrics_path = repo_root.join("models").join("PLANET_METAL_METRICS.json");
    std::fs::write(&metrics_path, serde_json::to_string_pretty(&metrics)?)?;
    println!("[planet_render] Saved {}", metrics_path.display());

    Ok(())
}
