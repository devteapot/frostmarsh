//! Wave 33B: Dual-Scale Locally-Flat Planetary Engine (`R = 1,300 m`, `6 × 2048² × 64` = `1.61B`
//! Virtual Voxels) — Apple M4 Pro Metal Visual & FPS Proof (`planet_horizon_demo`).
//!
//! Renders TWO synchronized planetary layers simultaneously using exact `f64` floating-origin
//! camera-relative transforms (`world_f64_to_camera_rel_f32`):
//! 1. **Layer 1 (Near `1 m` Detail Shard, `256×256×64` live `VolumeField`):**
//!    Curved `1 m` coastal terrain (golden beach, sapphire bay, emerald meadows, alpine cliffs),
//!    3D sculpted Nordic pine forests & coastal boulders, towering Orbital Floating Sky-Island
//!    with radiant cyan Buoyant Crystal Core & plunging waterfall!
//! 2. **Layer 2 (Global `R = 1,300 m` Macro Planet Shell, `6 × 2048² × 64`):**
//!    Full 6-face seamless macro continents, Fresnel sapphire oceans, polar glacier caps,
//!    glowing night-side equatorial volcanic rift, 12 orbiting Sky-Islands, and physical
//!    depth-aware Rayleigh+Mie atmosphere that smoothly transitions from a bright blue sunlit
//!    daytime sky at `2.5 m` eye-level to deep starry orbit at `3,300 m`!
//!
//! Outputs:
//! - `models/PLANET_HORIZON_METRICS.json`
//! - `models/planet_horizon_showcase/horizon_ground_1080p.png`
//! - `models/planet_horizon_showcase/horizon_low_flight_1080p.png`
//! - `models/planet_horizon_showcase/horizon_orbital_1080p.png`
//! - `models/planet_horizon_showcase/horizon_4view.png`

use bytemuck::{Pod, Zeroable};
use field_core::planet::{EDGE_COUNT, FACE_COUNT};
use field_core::planet_macro::{
    DualScalePlanet, MacroBiome, MacroPlanetConfig, MacroSkyIsland, PlanetShardKey, dot3_f64,
    face_uv_to_sphere_dir_f64, face_uv_z_to_world_f64, face_virtual_uv_f64, horizon_drop_meters,
    normalize3_f64, sample_macro_planet_column, sample_micro_relief_m, shard_neighbor,
    world_f64_to_camera_rel_f32,
};
use field_core::{SP_ICE, SP_PLANT, SP_ROCK, SP_SOIL, SP_WOOD};
use glam::{DVec3, Mat4, Vec3};
use image::{Rgba, RgbaImage};
use serde::Serialize;
use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

// =============================================================================
// Vertex & Floating-Origin Uniform Layouts
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

/// Camera-relative floating-origin uniforms (`view_proj` places camera at `(0, 0, 0)`!).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HorizonUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],      // planet-centered camera xyz (for sphere math/noise), time
    model_offset: [f32; 4], // (mesh_origin_f64 - cam_pos_f64) as f32, noise_freq_scale
    sun_dir: [f32; 4],      // xyz, r_ocean
    planet_cfg: [f32; 4],   // r_core, r_surf_atmo, r_atmo_outer, exposure
    viewport: [f32; 4],     // width, height, 1/width, 1/height
}

struct CameraShotF64 {
    pos: DVec3,
    target: DVec3,
    up: DVec3,
    fov_y_deg: f32,
    near: f32,
    far: f32,
    sun_dir: DVec3,
    exposure: f32,
}

impl CameraShotF64 {
    /// Floating-origin camera-relative view-projection matrix (camera sits strictly at `Vec3::ZERO`).
    fn view_proj_cam_rel(&self, w: u32, h: u32) -> (Mat4, Mat4) {
        let aspect = (w.max(1) as f32) / (h.max(1) as f32);
        let fwd_f64 = (self.target - self.pos).normalize();
        let fwd = Vec3::new(fwd_f64.x as f32, fwd_f64.y as f32, fwd_f64.z as f32);
        let up = Vec3::new(self.up.x as f32, self.up.y as f32, self.up.z as f32).normalize();
        let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, fwd, up);
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
// WGSL Shaders (Floating-Origin Terrain, Fresnel Ocean, Seamless Ground-to-Orbit
//               Rayleigh + Mie Atmosphere & ACES Filmic Tonemap)
// =============================================================================

const HORIZON_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    model_offset: vec4<f32>,
    sun_dir: vec4<f32>,
    planet_cfg: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

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

    var col = vec3<f32>(0.0025, 0.0045, 0.011);
    let band = exp(-pow(dot(rd, normalize(vec3<f32>(0.26, 0.86, -0.43))) * 4.4, 2.0));
    let neb = fbm3(rd * 4.5 + vec3<f32>(2.1, 0.7, -1.3));
    col += band * vec3<f32>(0.016, 0.024, 0.048) * smoothstep(0.32, 0.82, neb);
    col += band * vec3<f32>(0.020, 0.011, 0.032) * smoothstep(0.55, 0.90, fbm3(rd * 8.0 + 4.0));

    let g1 = rd * 250.0;
    let c1 = floor(g1);
    let f1 = fract(g1) - 0.5;
    let h1 = hash31(c1);
    if (h1 > 0.992) {
        let jitter = vec3<f32>(hash31(c1 + 7.1), hash31(c1 + 19.3), hash31(c1 + 37.9)) - 0.5;
        let d = length(f1 - jitter * 0.55);
        let b = smoothstep(0.32, 0.04, d) * (0.55 + 1.45 * fract(h1 * 137.0));
        let tint = mix(vec3<f32>(0.72, 0.84, 1.0), vec3<f32>(1.0, 0.90, 0.72), fract(h1 * 53.0));
        col += tint * b * 1.35;
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
        col += tint2 * b2 * 1.75;
    }

    let sun_cos = max(dot(rd, sun), 0.0);
    let sun_disc = smoothstep(0.9992, 0.99985, sun_cos);
    let sun_halo = pow(sun_cos, 64.0) * 0.55 + pow(sun_cos, 8.0) * 0.03;
    col += vec3<f32>(1.0, 0.95, 0.84) * sun_disc * 10.0 + vec3<f32>(1.0, 0.80, 0.54) * sun_halo;

    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Floating-Origin Curved Terrain, Pine Forests, Sky-Islands & Waterfalls
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
    @location(0) rel_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) emissive: vec3<f32>,
    @location(5) mat: vec4<f32>,
};

@vertex
fn vs_planet(in: VertIn) -> VertOut {
    var out: VertOut;
    // Exact camera-relative floating-origin position (camera is at Vec3::ZERO)
    let rel_pos = in.pos + uni.model_offset.xyz;
    out.clip_pos = uni.view_proj * vec4<f32>(rel_pos, 1.0);
    out.rel_pos = rel_pos;
    out.world_pos = rel_pos + uni.cam_pos.xyz;
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
    let view_dir = normalize(-in.rel_pos);
    let dist_cam = length(in.rel_pos);

    // Distance-adaptive seamless 3D micro-detail (crisp 1m detail up close, broad ridges from orbit!)
    let freq_near = 0.38;
    let freq_far = 0.045;
    let blend_far = smoothstep(140.0, 950.0, dist_cam);
    let f_cur = mix(freq_near, freq_far, blend_far);

    let micro = fbm3(p * f_cur);
    let micro_fine = noise3(p * (f_cur * 4.2));
    let is_foliage = step(0.84, in.mat.x) * step(in.color.g, 0.35) * step(in.color.r, 0.18);
    let detail_mod = mix(0.80 + 0.32 * micro + 0.10 * (micro_fine - 0.5), 0.88 + 0.24 * micro_fine, is_foliage);

    var n = normalize(in.normal);
    let eps = mix(0.28, 2.2, blend_far);
    let gx = fbm3((p + vec3<f32>(eps, 0.0, 0.0)) * f_cur) - fbm3((p - vec3<f32>(eps, 0.0, 0.0)) * f_cur);
    let gy = fbm3((p + vec3<f32>(0.0, eps, 0.0)) * f_cur) - fbm3((p - vec3<f32>(0.0, eps, 0.0)) * f_cur);
    let gz = fbm3((p + vec3<f32>(0.0, 0.0, eps)) * f_cur) - fbm3((p - vec3<f32>(0.0, 0.0, eps)) * f_cur);
    let grad = vec3<f32>(gx, gy, gz);
    let bump_strength = mix(0.75, 0.15, is_foliage) * (1.0 - 0.35 * blend_far);
    n = normalize(n - (grad - radial * dot(grad, radial)) * bump_strength);

    // Expose natural granite cliff rock on steep terrain slopes (skip emissives & tree canopies)
    let slope = clamp(1.0 - max(dot(n, radial), 0.0), 0.0, 1.0);
    let rock_col = vec3<f32>(0.31, 0.32, 0.36) * (0.85 + 0.30 * micro_fine);
    let is_emissive = step(0.12, length(in.emissive));
    let is_prop = step(0.5, in.mat.y); // metallic >= 0.5 marks 3D trees/boulders/waterfalls (keep exact vertex albedo!)
    let rock_blend = smoothstep(0.24, 0.52, slope) * (1.0 - is_emissive) * (1.0 - is_prop) * 0.68;
    let base_albedo = mix(in.color, rock_col, rock_blend);

    // Day/Night terminator illumination
    let term_raw = dot(radial, sun);
    let day_mask = smoothstep(-0.14, 0.22, term_raw);
    let twilight = exp(-pow((term_raw - 0.02) * 5.2, 2.0));

    let ndl = max(dot(n, sun), 0.0);
    let wrap_ndl = max((dot(n, sun) + 0.16) / 1.16, 0.0);

    let sun_col = mix(vec3<f32>(1.0, 0.48, 0.18), vec3<f32>(1.08, 1.00, 0.90), smoothstep(-0.04, 0.30, term_raw));
    let diffuse = sun_col * (ndl * 1.52 * day_mask + wrap_ndl * 0.30 * twilight);

    // Luminous azure sky hemisphere bounce on day side + subtle indigo night ambient
    let sky_up = 0.5 + 0.5 * dot(n, radial);
    let day_amb = mix(vec3<f32>(0.05, 0.08, 0.13), vec3<f32>(0.16, 0.29, 0.48), sky_up) * (0.18 + 0.82 * day_mask);
    let night_amb = vec3<f32>(0.009, 0.015, 0.030) * (1.0 - day_mask);

    let half_v = normalize(sun + view_dir);
    let rough = clamp(in.mat.x, 0.10, 0.96);
    let shininess = mix(96.0, 6.0, rough);
    let spec = pow(max(dot(n, half_v), 0.0), shininess) * mix(0.46, 0.04, rough) * day_mask;

    let fres = pow(1.0 - max(dot(n, view_dir), 0.0), 3.4);
    let rim_col = mix(vec3<f32>(0.12, 0.36, 0.78), vec3<f32>(0.95, 0.44, 0.16), twilight) * fres * (0.16 * day_mask + 0.24 * twilight);

    let albedo = base_albedo * detail_mod;
    let em_pulse = 0.86 + 0.30 * smoothstep(0.25, 0.85, micro);
    let emissive = in.emissive * em_pulse * mix(1.65, 1.05, day_mask);

    let lit = albedo * (diffuse + day_amb + night_amb) + sun_col * spec + rim_col + emissive;
    return vec4<f32>(lit, 1.0);
}

// -----------------------------------------------------------------------------
// Curved Specular/Fresnel Sapphire Ocean & Coastal Turquoise Shallows
// -----------------------------------------------------------------------------
@fragment
fn fs_ocean(in: VertOut) -> @location(0) vec4<f32> {
    let water_depth = in.mat.z;
    if (water_depth < 0.04) {
        discard;
    }
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(-in.rel_pos);
    let dist_cam = length(in.rel_pos);
    let blend_far = smoothstep(140.0, 950.0, dist_cam);

    let ice_frac = clamp(in.mat.w, 0.0, 1.0);
    let t = uni.cam_pos.w;

    let wf = mix(0.42, 0.055, blend_far);
    let w1 = fbm3(p * wf + vec3<f32>(t * 0.24, 0.0, -t * 0.18));
    let w2 = fbm3(p * (wf * 2.3) + vec3<f32>(-t * 0.20, t * 0.15, t * 0.22));
    let eps = mix(0.22, 1.8, blend_far);
    let gx = fbm3((p + vec3<f32>(eps, 0.0, 0.0)) * (wf * 1.4)) - fbm3((p - vec3<f32>(eps, 0.0, 0.0)) * (wf * 1.4));
    let gy = fbm3((p + vec3<f32>(0.0, eps, 0.0)) * (wf * 1.4)) - fbm3((p - vec3<f32>(0.0, eps, 0.0)) * (wf * 1.4));
    let gz = fbm3((p + vec3<f32>(0.0, 0.0, eps)) * (wf * 1.4)) - fbm3((p - vec3<f32>(0.0, 0.0, eps)) * (wf * 1.4));
    let wave_g = vec3<f32>(gx, gy, gz);
    let wave_t = wave_g - radial * dot(wave_g, radial);
    let n = normalize(radial - wave_t * mix(0.48, 0.14, ice_frac));

    let term_raw = dot(radial, sun);
    let day_mask = smoothstep(-0.14, 0.22, term_raw);
    let twilight = exp(-pow((term_raw - 0.02) * 5.2, 2.0));
    let sun_col = mix(vec3<f32>(1.0, 0.48, 0.18), vec3<f32>(1.06, 0.99, 0.91), smoothstep(-0.04, 0.30, term_raw));

    // Luminous turquoise coastal shallows -> rich royal sapphire -> deep abyssal navy
    let shallow_col = vec3<f32>(0.04, 0.52, 0.66);
    let mid_col = vec3<f32>(0.015, 0.17, 0.45);
    let deep_col = vec3<f32>(0.006, 0.048, 0.20);
    let d_norm = clamp(water_depth / 4.2, 0.0, 1.0);
    var water_col = mix(shallow_col, mid_col, smoothstep(0.0, 0.36, d_norm));
    water_col = mix(water_col, deep_col, smoothstep(0.32, 1.0, d_norm));

    // Crisp lapping surf foam along golden beaches
    let foam = smoothstep(0.65, 0.06, water_depth) * smoothstep(0.30, 0.72, w1 * 0.6 + w2 * 0.4);
    water_col = mix(water_col, vec3<f32>(0.88, 0.96, 1.0), foam * 0.72);

    let ice_col = vec3<f32>(0.78, 0.91, 0.99) * (0.90 + 0.14 * w1);
    let base_col = mix(water_col, ice_col, ice_frac);

    let ndv = max(dot(n, view_dir), 0.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 4.2);
    let sky_refl = mix(vec3<f32>(0.015, 0.03, 0.07), vec3<f32>(0.24, 0.56, 0.98), day_mask)
        + vec3<f32>(1.0, 0.48, 0.18) * twilight * 0.65;

    let half_v = normalize(sun + view_dir);
    let spec_sharp = pow(max(dot(n, half_v), 0.0), mix(180.0, 28.0, ice_frac)) * mix(2.4, 0.4, ice_frac);
    let spec_broad = pow(max(dot(n, half_v), 0.0), 22.0) * mix(0.28, 0.08, ice_frac);
    let spec = sun_col * (spec_sharp + spec_broad) * (day_mask + twilight * 0.45);

    let ndl = max(dot(n, sun), 0.0);
    let diff = base_col * (sun_col * ndl * 1.28 * day_mask + vec3<f32>(0.09, 0.18, 0.34) * (0.20 + 0.80 * day_mask) + vec3<f32>(0.008, 0.014, 0.032) * (1.0 - day_mask));

    let col = mix(diff, sky_refl, fresnel * mix(0.68, 0.22, ice_frac)) + spec;
    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Pass 2: Seamless Ground-to-Orbit Physical Rayleigh + Mie Atmosphere & ACES
// -----------------------------------------------------------------------------
@group(1) @binding(0) var hdr_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;

fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let r_ro = length(ro);
    let c = (r_ro - r) * (r_ro + r);
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
    let cam_alt = max(length(ro) - uni.planet_cfg.y, 0.0);
    let sun = normalize(uni.sun_dir.xyz);
    let r_surf = uni.planet_cfg.y; // ~1304.0m
    let r_atmo = uni.planet_cfg.z; // ~1520.0m
    let exposure = uni.planet_cfg.w;

    let rd = ray_dir_from_ndc(in.ndc);
    var t_scene = 1e6;
    let is_sky = select(0.0, 1.0, depth >= 0.99995);
    if (is_sky < 0.5) {
        // Unproject directly into camera-relative space (exact floating-origin distance!)
        let p_hit = uni.inv_view_proj * vec4<f32>(in.ndc, depth, 1.0);
        let rel_hit = p_hit.xyz / p_hit.w;
        t_scene = length(rel_hit);
    }

    let hit_atmo = intersect_sphere(ro, rd, r_atmo);
    if (hit_atmo.x < hit_atmo.y && hit_atmo.y > 0.0) {
        let t0 = max(hit_atmo.x, 0.0);
        let t1 = min(hit_atmo.y, t_scene);
        if (t1 > t0) {
            let steps = 18;
            let dt = (t1 - t0) / f32(steps);
            let h_atmo = max(r_atmo - r_surf, 10.0);
            let h_scale_r = h_atmo * 0.165; // ~35.6 m -> sleek vibrant limb with zero outer ring cutoff!
            let h_scale_m = h_atmo * 0.075; // ~16.2 m

            // Vivid deep Rayleigh blue & warm Mie horizon coefficients calibrated for R = 1,300 m
            let beta_r = vec3<f32>(0.0014, 0.0054, 0.0175);
            let beta_m = vec3<f32>(0.0010, 0.00088, 0.00076);

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
                // Smooth outer boundary taper so outer atmosphere rim fades invisibly into space!
                let outer_fade = smoothstep(r_atmo, r_atmo - h_atmo * 0.28, r_cur);

                let d_r = exp(-alt / h_scale_r) * outer_fade * dt;
                let d_m = exp(-alt / h_scale_m) * outer_fade * dt;
                od_r += d_r;
                od_m += d_m;

                let s_cos = dot(sp / max(r_cur, 1.0), sun);
                let sun_vis = smoothstep(-0.20, 0.18, s_cos);
                let sunset_od = mix(2.2, 0.14, smoothstep(-0.10, 0.45, s_cos));
                let tau_sun = beta_r * (sunset_od * 42.0) + beta_m * (sunset_od * 15.0);
                let tau_cam = beta_r * od_r + beta_m * od_m;
                let atten = exp(-(tau_cam + tau_sun)) * sun_vis;

                sum_r += d_r * atten;
                sum_m += d_m * atten;
            }

            // Smooth altitude transition:
            // - Ground / Low Flight (`cam_alt < 200 m`): rich sunlit azure daytime sky + crisp close terrain!
            // - Stratosphere / Orbit (`cam_alt > 260 m`): vivid sleek cobalt Rayleigh limb + crystal-clear continents!
            let in_atmo_factor = 1.0 - smoothstep(35.0, 240.0, cam_alt);
            let sky_gain = mix(1.05, 1.38, in_atmo_factor);
            // Keep near foreground (< 180m) ultra-crisp & vivid; build gentle blue aerial perspective toward distant horizon!
            let aerial_dist = smoothstep(45.0, 520.0, t_scene);
            let surf_gain = mix(0.14, 0.24 * aerial_dist, in_atmo_factor);
            let scatter_gain = mix(surf_gain, sky_gain, is_sky);

            let inscatter = (sum_r * beta_r * phase_r * 1.25 + sum_m * beta_m * phase_m * 0.34) * scatter_gain;
            let trans_surf = exp(-(beta_r * od_r * mix(0.10, 0.14 * aerial_dist, in_atmo_factor) + beta_m * od_m * 0.12));
            let trans_sky = exp(-(beta_r * od_r * 1.15 + beta_m * od_m * 0.90));

            // Suppress stars cleanly inside the sunlit daytime atmosphere (`cam_alt < 180 m`);
            // reveal full cosmic starfield smoothly in stratosphere (`600 m`) and orbit (`3,300 m`)!
            // Keep `bg_mult <= 1.0` so the outer `r_atmo` sphere boundary is 100% invisible in space!
            let sky_luma = dot(inscatter, vec3<f32>(0.2126, 0.7152, 0.0722));
            let day_cam = smoothstep(-0.05, 0.25, dot(normalize(ro), sun));
            let space_factor = smoothstep(130.0, 280.0, cam_alt);
            let star_keep = clamp(1.0 - sky_luma * 4.2, 0.0, 1.0) * mix(1.0 - day_cam, 1.0, space_factor);
            let bg_mult = mix(trans_surf, trans_sky * star_keep, is_sky);

            // Warm golden-amber twilight glow along the Day/Night terminator limb
            let p_mid = ro + rd * (0.5 * (t0 + t1));
            let mid_r = length(p_mid);
            let limb_f = exp(-max(mid_r - r_surf, 0.0) / (h_atmo * 0.20)) * smoothstep(r_atmo, r_atmo - h_atmo * 0.25, mid_r);
            let term_mid = dot(normalize(p_mid), sun);
            let dusk_glow = exp(-pow((term_mid - 0.0) * 5.2, 2.0)) * limb_f
                * vec3<f32>(1.0, 0.36, 0.09) * mix(0.08, 0.56, is_sky) * (1.0 - 0.6 * in_atmo_factor);

            // Soft sunlit cumulus high-cloud wisps visible from ground & low flight!
            var cloud_add = vec3<f32>(0.0);
            if (is_sky > 0.5 && in_atmo_factor > 0.05) {
                let hit_cloud = intersect_sphere(ro, rd, r_surf + 150.0);
                if (hit_cloud.y > 0.0) {
                    let tc = select(hit_cloud.y, hit_cloud.x, hit_cloud.x > 0.0);
                    if (tc < 950.0) {
                        let cp = ro + rd * tc;
                        let c_rad = normalize(cp);
                        let c_sun = smoothstep(-0.05, 0.35, dot(c_rad, sun));
                        let cn = fbm3(c_rad * 26.0 + vec3<f32>(1.7, -0.8, 2.3));
                        let c_mask = smoothstep(0.57, 0.82, cn) * smoothstep(950.0, 280.0, tc) * in_atmo_factor * c_sun;
                        cloud_add = vec3<f32>(0.90, 0.95, 1.00) * c_mask * 0.42;
                    }
                }
            }

            hdr = hdr * bg_mult + inscatter + dusk_glow + cloud_add;
        }
    }

    let mapped = aces_filmic(hdr * exposure);
    let srgb = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
"#;

// =============================================================================
// Dual-Scale Mesh Builders (Near 1m Shard + Global 1300m Macro Planet)
// =============================================================================

struct DualScaleMeshes {
    near_origin_f64: [f64; 3],
    near_terrain_verts: Vec<PlanetVertex>,
    near_terrain_indices: Vec<u32>,
    near_ocean_verts: Vec<PlanetVertex>,
    near_ocean_indices: Vec<u32>,
    macro_terrain_verts: Vec<PlanetVertex>,
    macro_terrain_indices: Vec<u32>,
    macro_ocean_verts: Vec<PlanetVertex>,
    macro_ocean_indices: Vec<u32>,
    sky_islands_verts: Vec<PlanetVertex>,
    sky_islands_indices: Vec<u32>,
}

fn cross_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale_f64(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn tangent_basis_f64(dir: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let ref_up = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t1 = normalize3_f64(cross_f64(ref_up, dir));
    let t2 = normalize3_f64(cross_f64(dir, t1));
    (t1, t2)
}

/// Emit a 3D sculpted Nordic pine tree (cedar trunk + 3 stacked tapered emerald foliage cones)
/// anchored cleanly along the local radial up direction `up_dir` on the sphere!
fn push_pine_tree(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up_dir: [f64; 3],
    origin_f64: [f64; 3],
    height_m: f64,
    radius_m: f64,
    tint_var: f32,
) {
    let (t1, t2) = tangent_basis_f64(up_dir);
    let mut emit_cone = |z0: f64,
                         z1: f64,
                         r0: f64,
                         r1: f64,
                         segs: usize,
                         col_bot: [f32; 3],
                         col_top: [f32; 3],
                         rough: f32| {
        let base_idx = verts.len() as u32;
        for s in 0..=segs {
            let ang = (s as f64 / segs as f64) * std::f64::consts::TAU;
            let ca = ang.cos();
            let sa = ang.sin();
            let rad_dir = add_f64(scale_f64(t1, ca), scale_f64(t2, sa));
            let n_dir = normalize3_f64(add_f64(scale_f64(rad_dir, 0.85), scale_f64(up_dir, 0.52)));
            let n_f32 = [n_dir[0] as f32, n_dir[1] as f32, n_dir[2] as f32];

            let p0 = add_f64(
                add_f64(base_world, scale_f64(up_dir, z0)),
                scale_f64(rad_dir, r0),
            );
            let p1 = add_f64(
                add_f64(base_world, scale_f64(up_dir, z1)),
                scale_f64(rad_dir, r1),
            );
            verts.push(PlanetVertex {
                pos: world_f64_to_camera_rel_f32(p0, origin_f64),
                normal: n_f32,
                color: col_bot,
                emissive: [0.0; 3],
                roughness: rough,
                metallic: 0.6, // prop flag: preserve crisp tree vertex color
                water_depth: 0.0,
                ice_frac: 0.0,
            });
            verts.push(PlanetVertex {
                pos: world_f64_to_camera_rel_f32(p1, origin_f64),
                normal: n_f32,
                color: col_top,
                emissive: [0.0; 3],
                roughness: rough,
                metallic: 0.6,
                water_depth: 0.0,
                ice_frac: 0.0,
            });
        }
        for s in 0..segs {
            let i0 = base_idx + (s * 2) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + 2;
            let i3 = i0 + 3;
            indices.extend_from_slice(&[i0, i2, i3, i0, i3, i1]);
        }
    };

    // Cedar trunk
    emit_cone(
        -0.4,
        height_m * 0.34,
        radius_m * 0.24,
        radius_m * 0.14,
        4,
        [0.18, 0.11, 0.06],
        [0.24, 0.15, 0.09],
        0.90,
    );

    // Three layered Nordic pine foliage tiers
    let g0 = [0.038 + 0.018 * tint_var, 0.16 + 0.05 * tint_var, 0.065];
    let g1 = [0.065 + 0.025 * tint_var, 0.27 + 0.07 * tint_var, 0.095];
    let g2 = [0.095 + 0.030 * tint_var, 0.34 + 0.06 * tint_var, 0.115];

    emit_cone(height_m * 0.22, height_m * 0.60, radius_m * 1.05, radius_m * 0.20, 5, g0, g1, 0.88);
    emit_cone(height_m * 0.45, height_m * 0.82, radius_m * 0.82, radius_m * 0.14, 5, g0, g1, 0.88);
    emit_cone(height_m * 0.68, height_m * 1.04, radius_m * 0.54, 0.04, 5, g1, g2, 0.86);
}

/// Emit a faceted coastal granite boulder for grounded `1 m` foreground scale!
fn push_boulder(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up_dir: [f64; 3],
    origin_f64: [f64; 3],
    rx: f64,
    ry: f64,
    rz: f64,
) {
    let (t1, t2) = tangent_basis_f64(up_dir);
    let segs = 7usize;
    let rings = 3usize;
    let base_idx = verts.len() as u32;
    let stride = segs + 1;
    for ri in 0..=rings {
        let phi = (ri as f64 / rings as f64) * std::f64::consts::PI - std::f64::consts::FRAC_PI_2;
        let cp = phi.cos();
        let sp = phi.sin();
        for si in 0..=segs {
            let th = (si as f64 / segs as f64) * std::f64::consts::TAU;
            let lx = cp * th.cos() * rx;
            let ly = cp * th.sin() * ry;
            let lz = (sp * 0.65 + 0.35) * rz;
            let p = add_f64(
                add_f64(base_world, scale_f64(up_dir, lz)),
                add_f64(scale_f64(t1, lx), scale_f64(t2, ly)),
            );
            let n = normalize3_f64(add_f64(
                scale_f64(up_dir, sp),
                add_f64(scale_f64(t1, cp * th.cos()), scale_f64(t2, cp * th.sin())),
            ));
            let shade = 0.28 + 0.14 * ((sp + 1.0) * 0.5) as f32;
            verts.push(PlanetVertex {
                pos: world_f64_to_camera_rel_f32(p, origin_f64),
                normal: [n[0] as f32, n[1] as f32, n[2] as f32],
                color: [shade, shade * 1.02, shade * 1.07],
                emissive: [0.0; 3],
                roughness: 0.78,
                metallic: 0.6,
                water_depth: 0.0,
                ice_frac: 0.0,
            });
        }
    }
    for ri in 0..rings {
        for si in 0..segs {
            let i00 = base_idx + (ri * stride + si) as u32;
            let i10 = i00 + 1;
            let i01 = base_idx + ((ri + 1) * stride + si) as u32;
            let i11 = i01 + 1;
            indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }
}

/// Emit one sculpted Floating Sky-Island (lush emerald plateau + sky-pines + tapered craggy
/// rocky underside + radiant glowing cyan-amethyst Buoyant Crystal Core + optional plunging waterfall!).
fn push_sky_island(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    isl: &MacroSkyIsland,
    r_core: f64,
    z_sea: f64,
    origin_f64: [f64; 3],
    waterfall_az: Option<f64>,
    add_sky_pines: bool,
) {
    let cdir = normalize3_f64(isl.center_dir);
    let (t1, t2) = tangent_basis_f64(cdir);
    let r_surf = r_core + z_sea;
    let rings = if add_sky_pines { 14usize } else { 8usize };
    let segs = if add_sky_pines { 36usize } else { 20usize };

    let dir_at_offset = |dx_m: f64, dy_m: f64| -> [f64; 3] {
        normalize3_f64(add_f64(
            cdir,
            add_f64(scale_f64(t1, dx_m / r_surf), scale_f64(t2, dy_m / r_surf)),
        ))
    };

    let eval_top = |t: f64, ang: f64| -> ([f64; 3], [f32; 3], [f32; 3], f32) {
        let rad = isl.organic_radius_m(ang) * t;
        let d = dir_at_offset(rad * ang.cos(), rad * ang.sin());
        let hill = (1.0 - t * t).powf(1.15) * 3.8
            + 0.75 * (ang * 4.0 + isl.phase).sin() * t * (1.0 - t);
        let z_f = isl.z_top + hill;
        let pos = scale_f64(d, r_core + z_f);
        let col = if t > 0.86 {
            [0.34, 0.35, 0.39] // Rocky cliff rim
        } else if t < 0.45 {
            [0.065, 0.24, 0.10] // Deep pine crown
        } else {
            [0.13, 0.39, 0.16] // Sunlit sky-meadow
        };
        (pos, col, [0.0; 3], 0.80)
    };

    let eval_bot = |t: f64, ang: f64| -> ([f64; 3], [f32; 3], [f32; 3], f32) {
        let rad = isl.organic_radius_m(ang) * t;
        let d = dir_at_offset(rad * ang.cos(), rad * ang.sin());
        let crag = 1.65 * (ang * 6.0 - isl.phase).sin() * t * (1.0 - t)
            + 0.85 * (ang * 11.0 + isl.phase * 1.7).cos() * t * (1.0 - t);
        let z_f = isl.z_bot + (isl.z_top - isl.z_bot) * t.powf(0.58) + crag;
        let pos = scale_f64(d, r_core + z_f);
        let core_glow = (1.0 - (t / 0.44).clamp(0.0, 1.0)).powf(1.65) as f32;
        // Glowing cyan-amethyst Buoyant Crystal Core veining up through dark sculpted basalt rock!
        let vein = ((ang * 5.0 + isl.phase).sin().abs().powf(8.0) as f32)
            * ((1.0 - t as f32).clamp(0.0, 1.0))
            * 0.65;
        let glow = (core_glow + vein * 0.42).clamp(0.0, 1.0);
        let rock_dark = 0.11 + 0.08 * (t as f32);
        let col = [
            rock_dark + 0.20 * glow,
            rock_dark * 1.06 + 0.66 * glow,
            rock_dark * 1.22 + 0.82 * glow,
        ];
        let em = [
            0.30 * glow,
            1.85 * glow,
            3.35 * glow,
        ];
        (pos, col, em, 0.76 - 0.50 * glow)
    };

    for is_top in [true, false] {
        let base_idx = verts.len() as u32;
        let stride_s = segs + 1;
        for ri in 0..=rings {
            let t = ri as f64 / rings as f64;
            for si in 0..=segs {
                let ang = (si as f64 / segs as f64) * std::f64::consts::TAU;
                let (pos, col, em, rough) = if is_top {
                    eval_top(t, ang)
                } else {
                    eval_bot(t, ang)
                };
                let dt = 0.03;
                let da = 0.06;
                let p_ta = if is_top {
                    eval_top((t + dt).min(1.0), ang).0
                } else {
                    eval_bot((t + dt).min(1.0), ang).0
                };
                let p_tb = if is_top {
                    eval_top((t - dt).max(0.0), ang).0
                } else {
                    eval_bot((t - dt).max(0.0), ang).0
                };
                let p_aa = if is_top {
                    eval_top(t.max(0.04), ang + da).0
                } else {
                    eval_bot(t.max(0.04), ang + da).0
                };
                let p_ab = if is_top {
                    eval_top(t.max(0.04), ang - da).0
                } else {
                    eval_bot(t.max(0.04), ang - da).0
                };
                let rad_dir = normalize3_f64(pos);
                let mut norm = normalize3_f64(cross_f64(sub_f64(p_ta, p_tb), sub_f64(p_aa, p_ab)));
                let want_sign = if is_top { 1.0 } else { -1.0 };
                if dot3_f64(norm, rad_dir) * want_sign < 0.0 {
                    norm = scale_f64(norm, -1.0);
                }
                verts.push(PlanetVertex {
                    pos: world_f64_to_camera_rel_f32(pos, origin_f64),
                    normal: [norm[0] as f32, norm[1] as f32, norm[2] as f32],
                    color: col,
                    emissive: em,
                    roughness: rough,
                    metallic: 0.6,
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
                    indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                } else {
                    indices.extend_from_slice(&[i00, i11, i10, i00, i01, i11]);
                }
            }
        }
    }

    // Crown the hero Sky-Island plateau with lush 3D sky-pines!
    if add_sky_pines {
        for pi in 0..22usize {
            let f_i = pi as f64;
            let ang = f_i * 2.39996 + isl.phase;
            let t_r = ((f_i + 0.5) / 22.0).sqrt() * 0.72;
            let (p_top, _, _, _) = eval_top(t_r, ang);
            let up_d = normalize3_f64(p_top);
            let h_tree = 5.5 + (pi % 4) as f64 * 1.4;
            let r_tree = 1.8 + (pi % 3) as f64 * 0.45;
            push_pine_tree(
                verts,
                indices,
                p_top,
                up_d,
                origin_f64,
                h_tree,
                r_tree,
                (pi % 5) as f32 * 0.2,
            );
        }
    }

    // Plunging luminous waterfall ribbon from Sky-Island rim straight into the sapphire bay!
    if let Some(wf_ang) = waterfall_az {
        let rim_r = isl.organic_radius_m(wf_ang) * 0.96;
        let out_dir = normalize3_f64(add_f64(
            scale_f64(t1, wf_ang.cos()),
            scale_f64(t2, wf_ang.sin()),
        ));
        let side_dir = normalize3_f64(cross_f64(cdir, out_dir));
        let z_start = isl.z_top + 0.2;
        let z_end = z_sea + 0.25;
        let wf_segs = 18usize;
        let wf_cols = 4usize;
        let base_idx = verts.len() as u32;
        let stride = wf_cols + 1;

        for ri in 0..=wf_segs {
            let s = ri as f64 / wf_segs as f64;
            let z_cur = z_start + (z_end - z_start) * (s * s * 0.65 + s * 0.35);
            let arc_out = rim_r + 3.8 * (s.sqrt());
            let half_w = 1.15 + 1.15 * s;
            for ci in 0..=wf_cols {
                let u_w = (ci as f64 / wf_cols as f64) * 2.0 - 1.0;
                let bow = (1.0 - u_w * u_w) * 0.65;
                let dx_m = out_dir[0] * (arc_out + bow) + side_dir[0] * (u_w * half_w);
                let dy_m = out_dir[1] * (arc_out + bow) + side_dir[1] * (u_w * half_w);
                let dz_m = out_dir[2] * (arc_out + bow) + side_dir[2] * (u_w * half_w);
                let d_cur = normalize3_f64([
                    cdir[0] + dx_m / r_surf,
                    cdir[1] + dy_m / r_surf,
                    cdir[2] + dz_m / r_surf,
                ]);
                let pos = scale_f64(d_cur, r_core + z_cur);
                let streak = 0.5 + 0.5 * ((u_w * 7.0 + s * 14.0).sin() as f32);
                let foam = (0.35 + 0.35 * s as f32 + 0.25 * streak).clamp(0.0, 1.0);
                let col = [
                    0.22 + 0.55 * foam,
                    0.62 + 0.32 * foam,
                    0.86 + 0.14 * foam,
                ];
                let em = [0.06 * foam, 0.20 * foam, 0.32 * foam];
                verts.push(PlanetVertex {
                    pos: world_f64_to_camera_rel_f32(pos, origin_f64),
                    normal: [out_dir[0] as f32, out_dir[1] as f32, out_dir[2] as f32],
                    color: col,
                    emissive: em,
                    roughness: 0.14,
                    metallic: 0.6,
                    water_depth: 0.0,
                    ice_frac: 0.0,
                });
            }
        }
        for ri in 0..wf_segs {
            for ci in 0..wf_cols {
                let i00 = base_idx + (ri * stride + ci) as u32;
                let i10 = i00 + 1;
                let i01 = base_idx + ((ri + 1) * stride + ci) as u32;
                let i11 = i01 + 1;
                // Double-sided waterfall ribbon so visible from all flight & ground angles!
                indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                indices.extend_from_slice(&[i00, i11, i10, i00, i01, i11]);
            }
        }

        // Splash mist ring on the bay surface right below the waterfall!
        let splash_r_base = rim_r + 3.2;
        let splash_dir = normalize3_f64([
            cdir[0] + out_dir[0] * splash_r_base / r_surf,
            cdir[1] + out_dir[1] * splash_r_base / r_surf,
            cdir[2] + out_dir[2] * splash_r_base / r_surf,
        ]);
        let (st1, st2) = tangent_basis_f64(splash_dir);
        let sp_base = verts.len() as u32;
        let sp_segs = 16usize;
        verts.push(PlanetVertex {
            pos: world_f64_to_camera_rel_f32(scale_f64(splash_dir, r_core + z_sea + 0.95), origin_f64),
            normal: [splash_dir[0] as f32, splash_dir[1] as f32, splash_dir[2] as f32],
            color: [0.88, 0.96, 1.0],
            emissive: [0.22, 0.38, 0.48],
            roughness: 0.25,
            metallic: 0.6,
            water_depth: 0.0,
            ice_frac: 0.0,
        });
        for si in 0..=sp_segs {
            let a = (si as f64 / sp_segs as f64) * std::f64::consts::TAU;
            let rad_m = 7.5 + 1.5 * (a * 5.0).sin();
            let d_ring = normalize3_f64(add_f64(
                splash_dir,
                add_f64(
                    scale_f64(st1, a.cos() * rad_m / r_surf),
                    scale_f64(st2, a.sin() * rad_m / r_surf),
                ),
            ));
            let pos = scale_f64(d_ring, r_core + z_sea + 0.45);
            verts.push(PlanetVertex {
                pos: world_f64_to_camera_rel_f32(pos, origin_f64),
                normal: [d_ring[0] as f32, d_ring[1] as f32, d_ring[2] as f32],
                color: [0.70, 0.88, 0.96],
                emissive: [0.06, 0.14, 0.20],
                roughness: 0.35,
                metallic: 0.6,
                water_depth: 0.0,
                ice_frac: 0.0,
            });
        }
        for si in 0..sp_segs {
            let i1 = sp_base + 1 + si as u32;
            let i2 = i1 + 1;
            indices.extend_from_slice(&[sp_base, i1, i2, sp_base, i2, i1]);
        }
    }
}

/// Build BOTH synchronized planetary mesh layers:
/// - **Near `1 m` Detail Layer** (`256×256` live hydrated `VolumeField` shard + 3D pine forests +
///   coastal boulders + near ocean + edge skirts) anchored at `near_origin_f64`
/// - **Global `R = 1,300 m` Macro Planet Layer** (`6 × 160 × 160` curved continents + Fresnel
///   sapphire ocean + 12 Orbital Sky-Islands), cleanly clipped inside the hydrated shard footprint
///   to prevent any Z-fighting or seam cracks!
fn build_dual_scale_meshes(
    planet: &DualScalePlanet,
    hero_key: PlanetShardKey,
    waterfall_az: f64,
) -> DualScaleMeshes {
    let cfg = planet.config;
    let r_core = cfg.r_core;
    let z_sea = cfg.z_sea as f64;
    let z_sea_surf = z_sea + 0.55;
    let n_virt = cfg.face_size_virtual();
    let s = cfg.shard_size;
    let seed = cfg.seed;

    let hero_vf = planet
        .shards
        .get(&hero_key)
        .expect("hero shard must be hydrated before meshing");

    // Exact `f64` anchor origin for the near `1 m` shard (center of shard at sea level)
    let hero_dir_center = hero_key.center_sphere_dir(cfg.shards_per_face_axis);
    let near_origin_f64 = scale_f64(hero_dir_center, r_core + z_sea_surf);

    // -------------------------------------------------------------------------
    // LAYER 1: Near `1 m` Detail Terrain + 3D Pine Forests + Coastal Boulders
    // -------------------------------------------------------------------------
    let stride = s + 1;
    let mut near_terrain_verts = Vec::with_capacity(stride * stride + 60_000);
    let mut near_terrain_indices = Vec::with_capacity(s * s * 6 + 120_000);
    let mut near_ocean_verts = Vec::with_capacity(stride * stride);
    let mut near_ocean_indices = Vec::with_capacity(s * s * 6);

    // Precompute exact continuous surface height & live VolumeField species tint per vertex
    let mut grid_pos: Vec<[f64; 3]> = Vec::with_capacity(stride * stride);
    let mut grid_dir: Vec<[f64; 3]> = Vec::with_capacity(stride * stride);
    let mut grid_surf_z: Vec<f64> = Vec::with_capacity(stride * stride);
    let mut grid_col: Vec<[f32; 3]> = Vec::with_capacity(stride * stride);
    let mut grid_em: Vec<[f32; 3]> = Vec::with_capacity(stride * stride);
    let mut grid_rough: Vec<f32> = Vec::with_capacity(stride * stride);
    let mut grid_wdepth: Vec<f32> = Vec::with_capacity(stride * stride);
    let mut grid_ice: Vec<f32> = Vec::with_capacity(stride * stride);

    let gx_base = hero_key.sx as usize * s;
    let gy_base = hero_key.sy as usize * s;

    for vy in 0..=s {
        let gy = (gy_base + vy) as f64;
        let ly = vy.min(s - 1);
        for vx in 0..=s {
            let gx = (gx_base + vx) as f64;
            let lx = vx.min(s - 1);
            let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
            let dir = face_uv_to_sphere_dir_f64(hero_key.face as usize, u, v);
            let col = sample_macro_planet_column(dir, seed);

            // Smoothly taper `1 m` micro-relief to `0.0` at the outer `10 m` rim of the shard so
            // the hydrated shard stitches into the Macro Planet Shell with zero step crack!
            let edge_d = vx.min(s - vx).min(vy).min(s - vy) as f64;
            let edge_blend = (edge_d / 10.0).clamp(0.0, 1.0);
            let micro = sample_micro_relief_m(dir, seed) * edge_blend * 0.85;
            let surf_z = (col.surf_z + micro).clamp(16.0, 42.5);

            // Blend live `VolumeField` surface material (soil/plant/wood/rock/ice) into albedo!
            let idx = ly * s + lx;
            let z_v = (surf_z.round() as usize).clamp(1, hero_vf.depth - 2);
            let l_v = &hero_vf.layers[z_v];
            let has_ice = l_v.mass[SP_ICE][idx] > 300;
            let has_wood = l_v.mass[SP_WOOD][idx] > 280;
            let has_plant = l_v.mass[SP_PLANT][idx] > 180;
            let has_rock = l_v.mass[SP_ROCK][idx] > 700 && l_v.mass[SP_SOIL][idx] < 200;

            let mut albedo = col.albedo;
            let mut rough = col.roughness;
            if has_ice {
                albedo = [0.85, 0.91, 0.97];
                rough = 0.25;
            } else if surf_z > z_sea_surf && surf_z < z_sea_surf + 1.25 {
                // Warm sunlit golden sandy coastline
                albedo = [0.54, 0.44, 0.27];
                rough = 0.88;
            } else if has_wood {
                albedo = [0.040, 0.165, 0.065];
                rough = 0.86;
            } else if has_plant && surf_z > z_sea_surf + 1.1 {
                let alt_t = ((surf_z - z_sea_surf) / 11.0).clamp(0.0, 1.0) as f32;
                albedo = [0.085 + 0.10 * alt_t, 0.28 - 0.05 * alt_t, 0.08 + 0.025 * alt_t];
                rough = 0.82;
            } else if has_rock && surf_z > z_sea_surf + 6.0 {
                albedo = [0.32, 0.34, 0.38];
                rough = 0.74;
            }

            let pos = scale_f64(dir, r_core + surf_z);
            let water_depth = (z_sea_surf - surf_z).max(0.0) as f32;
            let ice_frac = if water_depth > 0.05 && hero_vf.layers[cfg.z_sea.min(hero_vf.depth - 1)].mass[SP_ICE][idx] > 250 {
                0.75
            } else {
                0.0
            };

            grid_pos.push(pos);
            grid_dir.push(dir);
            grid_surf_z.push(surf_z);
            grid_col.push(albedo);
            grid_em.push(col.emissive);
            grid_rough.push(rough);
            grid_wdepth.push(water_depth);
            grid_ice.push(ice_frac);
        }
    }

    // Compute smooth central-difference normals and emit `1 m` terrain + ocean vertices
    for vy in 0..=s {
        for vx in 0..=s {
            let vi = vy * stride + vx;
            let xm = vx.saturating_sub(1);
            let xp = (vx + 1).min(s);
            let ym = vy.saturating_sub(1);
            let yp = (vy + 1).min(s);
            let dx = sub_f64(grid_pos[vy * stride + xp], grid_pos[vy * stride + xm]);
            let dy = sub_f64(grid_pos[yp * stride + vx], grid_pos[ym * stride + vx]);
            let mut n = normalize3_f64(cross_f64(dx, dy));
            if dot3_f64(n, grid_dir[vi]) < 0.0 {
                n = scale_f64(n, -1.0);
            }
            near_terrain_verts.push(PlanetVertex {
                pos: world_f64_to_camera_rel_f32(grid_pos[vi], near_origin_f64),
                normal: [n[0] as f32, n[1] as f32, n[2] as f32],
                color: grid_col[vi],
                emissive: grid_em[vi],
                roughness: grid_rough[vi],
                metallic: 0.0,
                water_depth: grid_wdepth[vi],
                ice_frac: grid_ice[vi],
            });

            let ocean_pos = scale_f64(grid_dir[vi], r_core + z_sea_surf);
            near_ocean_verts.push(PlanetVertex {
                pos: world_f64_to_camera_rel_f32(ocean_pos, near_origin_f64),
                normal: [
                    grid_dir[vi][0] as f32,
                    grid_dir[vi][1] as f32,
                    grid_dir[vi][2] as f32,
                ],
                color: [0.04, 0.24, 0.54],
                emissive: [0.0; 3],
                roughness: 0.08,
                metallic: 0.1,
                water_depth: grid_wdepth[vi],
                ice_frac: grid_ice[vi],
            });
        }
    }

    for vy in 0..s {
        for vx in 0..s {
            let i00 = (vy * stride + vx) as u32;
            let i10 = i00 + 1;
            let i01 = ((vy + 1) * stride + vx) as u32;
            let i11 = i01 + 1;
            near_terrain_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
            if grid_wdepth[i00 as usize] > 0.01
                || grid_wdepth[i10 as usize] > 0.01
                || grid_wdepth[i01 as usize] > 0.01
                || grid_wdepth[i11 as usize] > 0.01
            {
                near_ocean_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
            }
        }
    }

    // Subtle 4 m radial skirt around the 4 outer edges of the `1 m` shard (eliminates any grazing cracks!)
    let mut emit_skirt_edge = |coords: Vec<(usize, usize)>, reverse: bool| {
        let base = near_terrain_verts.len() as u32;
        for &(vx, vy) in &coords {
            let vi = vy * stride + vx;
            let top_v = near_terrain_verts[vi];
            let bot_pos = scale_f64(grid_dir[vi], r_core + grid_surf_z[vi] - 4.5);
            let mut bot_v = top_v;
            bot_v.pos = world_f64_to_camera_rel_f32(bot_pos, near_origin_f64);
            near_terrain_verts.push(top_v);
            near_terrain_verts.push(bot_v);
        }
        for i in 0..coords.len().saturating_sub(1) {
            let a = base + (i * 2) as u32;
            let b = a + 1;
            let c = a + 2;
            let d = a + 3;
            if reverse {
                near_terrain_indices.extend_from_slice(&[a, c, d, a, d, b]);
            } else {
                near_terrain_indices.extend_from_slice(&[a, d, c, a, b, d]);
            }
        }
    };
    emit_skirt_edge((0..=s).map(|vx| (vx, 0)).collect(), false);
    emit_skirt_edge((0..=s).map(|vx| (vx, s)).collect(), true);
    emit_skirt_edge((0..=s).map(|vy| (0, vy)).collect(), true);
    emit_skirt_edge((0..=s).map(|vy| (s, vy)).collect(), false);

    // Populate sculpted 3D Nordic pine trees & coastal granite boulders directly from the hydrated shard!
    for ly in (3..s - 3).step_by(4) {
        for lx in (3..s - 3).step_by(4) {
            let h = ((lx.wrapping_mul(73_856_093)) ^ (ly.wrapping_mul(19_349_663)) ^ (seed as usize)) % 1000;
            let jx = (lx + (h % 3)).min(s - 2);
            let jy = (ly + ((h / 7) % 3)).min(s - 2);
            let vi = jy * stride + jx;
            let sz = grid_surf_z[vi];
            let idx = jy * s + jx;
            let zv = (sz.round() as usize).clamp(1, hero_vf.depth - 2);
            let has_canopy = hero_vf.layers[zv].mass[SP_WOOD][idx] > 140
                || hero_vf.layers[zv].mass[SP_PLANT][idx] > 160
                || (zv + 1 < hero_vf.depth && hero_vf.layers[zv + 1].mass[SP_WOOD][idx] > 140);

            if sz > z_sea_surf + 1.35 && sz < 35.5 && has_canopy && (h % 100) < 52 {
                let tree_h = 5.4 + ((h % 37) as f64 / 37.0) * 4.6;
                let tree_r = 1.7 + ((h % 19) as f64 / 19.0) * 1.15;
                let tint = ((h % 11) as f32) / 11.0;
                push_pine_tree(
                    &mut near_terrain_verts,
                    &mut near_terrain_indices,
                    grid_pos[vi],
                    grid_dir[vi],
                    near_origin_f64,
                    tree_h,
                    tree_r,
                    tint,
                );
            } else if sz > z_sea_surf + 0.15 && sz < z_sea_surf + 3.2 && (h % 100) < 9 {
                let r_b = 0.85 + ((h % 13) as f64 / 13.0) * 1.25;
                push_boulder(
                    &mut near_terrain_verts,
                    &mut near_terrain_indices,
                    grid_pos[vi],
                    grid_dir[vi],
                    near_origin_f64,
                    r_b * 1.2,
                    r_b * 0.9,
                    r_b * 0.65,
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // LAYER 2: Global `R = 1,300 m` Macro Planet Shell (`6 × 144 × 144` quads)
    //          Anchored at planet center `[0, 0, 0]`, cleanly clipped inside the
    //          hydrated hero shard footprint to eliminate ALL Z-fighting!
    // -------------------------------------------------------------------------
    let macro_origin_f64 = [0.0f64; 3];
    let sub = 144usize;
    let m_stride = sub + 1;
    let mut macro_terrain_verts = Vec::with_capacity(FACE_COUNT * m_stride * m_stride);
    let mut macro_terrain_indices = Vec::with_capacity(FACE_COUNT * sub * sub * 6);
    let mut macro_ocean_verts = Vec::with_capacity(FACE_COUNT * m_stride * m_stride);
    let mut macro_ocean_indices = Vec::with_capacity(FACE_COUNT * sub * sub * 6);

    let (hu0, hu1, hv0, hv1) = hero_key.uv_bounds(cfg.shards_per_face_axis);
    let inset_uv = 0.008f64; // Tiny overlap margin so the 1m shard seamlessly covers the macro hole!

    for f in 0..FACE_COUNT {
        let base_t = macro_terrain_verts.len() as u32;
        let base_o = macro_ocean_verts.len() as u32;

        for iy in 0..=sub {
            let v = (iy as f64 / sub as f64) * 2.0 - 1.0;
            for ix in 0..=sub {
                let u = (ix as f64 / sub as f64) * 2.0 - 1.0;
                let dir = face_uv_to_sphere_dir_f64(f, u, v);
                let col = sample_macro_planet_column(dir, seed);

                // Smoothly sink overlap ring slightly inside the hydrated hero shard so near 1m mesh always wins cleanly
                let mut r_offset = 0.0f64;
                if f == hero_key.face as usize && u > hu0 && u < hu1 && v > hv0 && v < hv1 {
                    let du = (u - hu0).min(hu1 - u);
                    let dv = (v - hv0).min(hv1 - v);
                    let d_in = du.min(dv);
                    r_offset = -(d_in / inset_uv).clamp(0.0, 1.0) * 1.4;
                }

                let r = r_core + col.surf_z + r_offset;
                let pos = scale_f64(dir, r);

                // Seamless 3D spherical analytical gradient normal across all 12 cube seams!
                let (t1, t2) = tangent_basis_f64(dir);
                let eps = 0.0055f64;
                let d_p1 = normalize3_f64(add_f64(dir, scale_f64(t1, eps)));
                let d_m1 = normalize3_f64(sub_f64(dir, scale_f64(t1, eps)));
                let d_p2 = normalize3_f64(add_f64(dir, scale_f64(t2, eps)));
                let d_m2 = normalize3_f64(sub_f64(dir, scale_f64(t2, eps)));

                let r_p1 = r_core + sample_macro_planet_column(d_p1, seed).surf_z;
                let r_m1 = r_core + sample_macro_planet_column(d_m1, seed).surf_z;
                let r_p2 = r_core + sample_macro_planet_column(d_p2, seed).surf_z;
                let r_m2 = r_core + sample_macro_planet_column(d_m2, seed).surf_z;

                let dp1 = sub_f64(scale_f64(d_p1, r_p1), scale_f64(d_m1, r_m1));
                let dp2 = sub_f64(scale_f64(d_p2, r_p2), scale_f64(d_m2, r_m2));
                let mut norm = normalize3_f64(cross_f64(dp1, dp2));
                if dot3_f64(norm, dir) < 0.0 {
                    norm = scale_f64(norm, -1.0);
                }

                let water_depth = (z_sea_surf - col.surf_z).max(0.0) as f32;
                let ice_frac = if water_depth > 0.05 && col.temp_k < 271.5 {
                    ((271.5 - col.temp_k) / 8.0).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };

                macro_terrain_verts.push(PlanetVertex {
                    pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
                    normal: [norm[0] as f32, norm[1] as f32, norm[2] as f32],
                    color: col.albedo,
                    emissive: col.emissive,
                    roughness: col.roughness,
                    metallic: 0.0,
                    water_depth,
                    ice_frac,
                });

                let o_pos = scale_f64(dir, r_core + z_sea_surf + r_offset * 0.5);
                macro_ocean_verts.push(PlanetVertex {
                    pos: [o_pos[0] as f32, o_pos[1] as f32, o_pos[2] as f32],
                    normal: [dir[0] as f32, dir[1] as f32, dir[2] as f32],
                    color: [0.04, 0.24, 0.54],
                    emissive: [0.0; 3],
                    roughness: 0.08,
                    metallic: 0.1,
                    water_depth,
                    ice_frac,
                });
            }
        }

        for iy in 0..sub {
            let v_mid = ((iy as f64 + 0.5) / sub as f64) * 2.0 - 1.0;
            for ix in 0..sub {
                let u_mid = ((ix as f64 + 0.5) / sub as f64) * 2.0 - 1.0;
                // Clip macro quads strictly inside the inset hydrated hero shard footprint!
                let inside_hero = f == hero_key.face as usize
                    && u_mid > hu0 + inset_uv
                    && u_mid < hu1 - inset_uv
                    && v_mid > hv0 + inset_uv
                    && v_mid < hv1 - inset_uv;
                if inside_hero {
                    continue;
                }
                let i00 = base_t + (iy * m_stride + ix) as u32;
                let i10 = i00 + 1;
                let i01 = base_t + ((iy + 1) * m_stride + ix) as u32;
                let i11 = i01 + 1;
                macro_terrain_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);

                let o00 = base_o + (iy * m_stride + ix) as u32;
                let o10 = o00 + 1;
                let o01 = base_o + ((iy + 1) * m_stride + ix) as u32;
                let o11 = o01 + 1;
                macro_ocean_indices.extend_from_slice(&[o00, o10, o11, o00, o11, o01]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // LAYER 3: Hero Floating Sky-Island (in near floating-origin frame!) +
    //          Sister Orbital Sky-Islands around the R = 1,300 m globe
    // -------------------------------------------------------------------------
    // Put the hero Sky-Island right inside `near_terrain_verts` so it shares the exact sub-mm
    // floating-origin frame of the near shard! Put all sister Sky-Islands in `sky_islands_verts`.
    let mut sky_islands_verts = Vec::new();
    let mut sky_islands_indices = Vec::new();
    for (idx_i, isl) in planet.sky_islands.iter().enumerate() {
        if idx_i == 0 {
            push_sky_island(
                &mut near_terrain_verts,
                &mut near_terrain_indices,
                isl,
                r_core,
                z_sea_surf,
                near_origin_f64,
                Some(waterfall_az),
                true,
            );
        } else {
            push_sky_island(
                &mut sky_islands_verts,
                &mut sky_islands_indices,
                isl,
                r_core,
                z_sea_surf,
                macro_origin_f64,
                if isl.has_waterfall { Some(0.5) } else { None },
                false,
            );
        }
    }

    DualScaleMeshes {
        near_origin_f64,
        near_terrain_verts,
        near_terrain_indices,
        near_ocean_verts,
        near_ocean_indices,
        macro_terrain_verts,
        macro_terrain_indices,
        macro_ocean_verts,
        macro_ocean_indices,
        sky_islands_verts,
        sky_islands_indices,
    }
}

// =============================================================================
// Scenic Coastal Hero Shard & Camera Framing Solver
// =============================================================================

struct HeroSite {
    shard_key: PlanetShardKey,
    beach_dir: [f64; 3],
    beach_surf_z: f64,
    look_fwd: [f64; 3],
    island_dir: [f64; 3],
    waterfall_az: f64,
}

fn find_scenic_coastal_hero_site(cfg: MacroPlanetConfig, sun_dir: [f64; 3]) -> HeroSite {
    let m = cfg.shards_per_face_axis;
    let n_virt = cfg.face_size_virtual();
    let s = cfg.shard_size;
    let r_surf = cfg.surface_radius_m();

    let mut best_shard = PlanetShardKey::new(4, 3, 3);
    let mut best_shard_score = f64::NEG_INFINITY;

    for f in 0..FACE_COUNT as u8 {
        for sy in 0..m {
            for sx in 0..m {
                let key = PlanetShardKey::new(f, sx, sy);
                let cdir = key.center_sphere_dir(m);
                let sdot = dot3_f64(cdir, sun_dir);
                if sdot < 0.56 || cdir[1].abs() > 0.42 {
                    continue;
                }
                let mut water_cnt = 0usize;
                let mut land_cnt = 0usize;
                let mut forest_cnt = 0usize;
                let mut max_z = 0.0f64;
                let mut has_rift = false;
                for iy in 0..8 {
                    let gy = (sy as usize * s) as f64 + (iy as f64 + 0.5) * (s as f64 / 8.0);
                    for ix in 0..8 {
                        let gx = (sx as usize * s) as f64 + (ix as f64 + 0.5) * (s as f64 / 8.0);
                        let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
                        let dir = face_uv_to_sphere_dir_f64(f as usize, u, v);
                        let col = sample_macro_planet_column(dir, cfg.seed);
                        if col.is_rift_core || col.is_rift_rim || col.is_polar {
                            has_rift = true;
                        }
                        if col.water_depth > 0.3 {
                            water_cnt += 1;
                        } else {
                            land_cnt += 1;
                        }
                        if col.biome == MacroBiome::PineForest || col.biome == MacroBiome::EmeraldMeadow {
                            forest_cnt += 1;
                        }
                        if col.surf_z > max_z {
                            max_z = col.surf_z;
                        }
                    }
                }
                if has_rift || water_cnt < 14 || land_cnt < 22 {
                    continue;
                }
                let frac_w = water_cnt as f64 / 64.0;
                let coast_balance = 1.0 - (frac_w - 0.42).abs() * 3.0;
                let relief = ((max_z - 25.0) / 14.0).clamp(0.0, 1.4);
                let score = coast_balance * 4.0 + relief * 2.5 + (forest_cnt as f64 / 64.0) * 2.0 + sdot * 2.5;
                if score > best_shard_score {
                    best_shard_score = score;
                    best_shard = key;
                }
            }
        }
    }

    // Search inside `best_shard` for an unobstructed sunlit golden beach / coastal meadow vantage
    // looking out across the sapphire bay toward forested hills & the open horizon!
    let gx0 = best_shard.sx as usize * s;
    let gy0 = best_shard.sy as usize * s;

    let mut best_cam_score = f64::NEG_INFINITY;
    let mut best_beach_dir = best_shard.center_sphere_dir(m);
    let mut best_beach_z = 26.0;
    let mut best_fwd = [1.0, 0.0, 0.0];

    for ly in (28..s - 28).step_by(12) {
        for lx in (28..s - 28).step_by(12) {
            let (u, v) = face_virtual_uv_f64((gx0 + lx) as f64 + 0.5, (gy0 + ly) as f64 + 0.5, n_virt);
            let dir = face_uv_to_sphere_dir_f64(best_shard.face as usize, u, v);
            let col = sample_macro_planet_column(dir, cfg.seed);
            // Ideal beach / low coastal meadow elevation (`25.0 .. 28.2`)
            if col.surf_z < 24.9 || col.surf_z > 28.5 {
                continue;
            }
            let (t1, t2) = tangent_basis_f64(dir);
            for dir_i in 0..12 {
                let ang = (dir_i as f64 / 12.0) * std::f64::consts::TAU;
                let fwd = normalize3_f64(add_f64(scale_f64(t1, ang.cos()), scale_f64(t2, ang.sin())));

                // Check foreground beach/slope, midground sapphire bay (30..105m), and distant scenic coast/ocean!
                let mut bay_water = 0.0f64;
                let mut blocked_fg = false;
                for &d_m in &[12.0f64, 24.0, 42.0, 62.0, 82.0, 102.0, 130.0, 175.0, 240.0] {
                    let pd = normalize3_f64(add_f64(dir, scale_f64(fwd, d_m / r_surf)));
                    let pc = sample_macro_planet_column(pd, cfg.seed);
                    if d_m <= 28.0 && pc.surf_z > col.surf_z + 1.8 {
                        blocked_fg = true;
                        break;
                    }
                    if (35.0..=115.0).contains(&d_m) && pc.water_depth > 0.3 {
                        bay_water += 1.0;
                    }
                }
                if blocked_fg {
                    continue;
                }
                // Check flanking forested coast to the left/right at 35..80m so foreground feels lush & framed
                let right = normalize3_f64(cross_f64(fwd, dir));
                let mut flank_land = 0.0f64;
                for &side in &[-28.0f64, 28.0, -48.0, 48.0] {
                    let pd = normalize3_f64(add_f64(
                        dir,
                        add_f64(scale_f64(fwd, 40.0 / r_surf), scale_f64(right, side / r_surf)),
                    ));
                    let pc = sample_macro_planet_column(pd, cfg.seed);
                    if pc.surf_z > 25.5 && pc.surf_z < 36.0 {
                        flank_land += 1.0;
                    }
                }
                // Prefer side-lit sun (~40°-110° off forward) for sculpted shadows & sparkling ocean glint!
                let sun_align = dot3_f64(fwd, sun_dir);
                let light_score = 1.0 - (sun_align - 0.15).abs();
                let sc = bay_water * 2.2 + flank_land * 1.1 + light_score * 1.8;
                if sc > best_cam_score {
                    best_cam_score = sc;
                    best_beach_dir = dir;
                    best_beach_z = col.surf_z;
                    best_fwd = fwd;
                }
            }
        }
    }

    // Place the Hero Floating Sky-Island `94 m` ahead and `14 m` to the right over the bay so it
    // towers majestically in the upper-right third of the ground vista with its waterfall facing us!
    let right = normalize3_f64(cross_f64(best_fwd, best_beach_dir));
    let island_dir = normalize3_f64(add_f64(
        best_beach_dir,
        add_f64(scale_f64(best_fwd, 94.0 / r_surf), scale_f64(right, 14.0 / r_surf)),
    ));

    // Compute exact azimuth on the Sky-Island pointing back toward the viewer so the plunging
    // waterfall cascades right into our field of view!
    let (it1, it2) = tangent_basis_f64(island_dir);
    let to_viewer = sub_f64(best_beach_dir, island_dir);
    let wf_az = dot3_f64(to_viewer, it2).atan2(dot3_f64(to_viewer, it1)) - 0.22;

    HeroSite {
        shard_key: best_shard,
        beach_dir: best_beach_dir,
        beach_surf_z: best_beach_z,
        look_fwd: best_fwd,
        island_dir,
        waterfall_az: wf_az,
    }
}

// =============================================================================
// Headless Apple M4 Pro Metal Dual-Scale Renderer
// =============================================================================

struct MetalHorizonRenderer {
    adapter_name: String,
    adapter_backend: String,
    device: wgpu::Device,
    queue: wgpu::Queue,

    uni_near_buf: wgpu::Buffer,
    bg_uni_near: wgpu::BindGroup,
    uni_macro_buf: wgpu::Buffer,
    bg_uni_macro: wgpu::BindGroup,
    resolve_bgl: wgpu::BindGroupLayout,

    pipe_starfield: wgpu::RenderPipeline,
    pipe_terrain: wgpu::RenderPipeline,
    pipe_ocean: wgpu::RenderPipeline,
    pipe_resolve: wgpu::RenderPipeline,

    near_origin_f64: [f64; 3],
    near_t_vbuf: wgpu::Buffer,
    near_t_ibuf: wgpu::Buffer,
    near_t_icount: u32,
    near_o_vbuf: wgpu::Buffer,
    near_o_ibuf: wgpu::Buffer,
    near_o_icount: u32,

    macro_t_vbuf: wgpu::Buffer,
    macro_t_ibuf: wgpu::Buffer,
    macro_t_icount: u32,
    macro_o_vbuf: wgpu::Buffer,
    macro_o_ibuf: wgpu::Buffer,
    macro_o_icount: u32,
    sky_vbuf: wgpu::Buffer,
    sky_ibuf: wgpu::Buffer,
    sky_icount: u32,

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
        label: Some("horizon_hdr"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("horizon_depth"),
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
        label: Some("horizon_resolve_bg"),
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

impl MetalHorizonRenderer {
    async fn new(meshes: &DualScaleMeshes, cfg: MacroPlanetConfig) -> Result<Self, Box<dyn std::error::Error>> {
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
                label: Some("horizon_metal_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("horizon_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(HORIZON_SHADER_WGSL)),
        });

        let uni_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("horizon_uni_bgl"),
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
            label: Some("horizon_resolve_bgl"),
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

        let uni_size = std::mem::size_of::<HorizonUniforms>() as u64;
        let uni_near_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni_near_buf"),
            size: uni_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_uni_near = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_uni_near"),
            layout: &uni_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uni_near_buf.as_entire_binding(),
            }],
        });
        let uni_macro_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni_macro_buf"),
            size: uni_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_uni_macro = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_uni_macro"),
            layout: &uni_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uni_macro_buf.as_entire_binding(),
            }],
        });

        let hdr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("horizon_hdr_pl"),
            bind_group_layouts: &[Some(&uni_bgl)],
            immediate_size: 0,
        });
        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("horizon_resolve_pl"),
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

        let near_t_vbuf = create_buf(&device, &queue, "near_t_vbuf", &meshes.near_terrain_verts, wgpu::BufferUsages::VERTEX);
        let near_t_ibuf = create_buf(&device, &queue, "near_t_ibuf", &meshes.near_terrain_indices, wgpu::BufferUsages::INDEX);
        let near_o_vbuf = create_buf(&device, &queue, "near_o_vbuf", &meshes.near_ocean_verts, wgpu::BufferUsages::VERTEX);
        let near_o_ibuf = create_buf(&device, &queue, "near_o_ibuf", &meshes.near_ocean_indices, wgpu::BufferUsages::INDEX);

        let macro_t_vbuf = create_buf(&device, &queue, "macro_t_vbuf", &meshes.macro_terrain_verts, wgpu::BufferUsages::VERTEX);
        let macro_t_ibuf = create_buf(&device, &queue, "macro_t_ibuf", &meshes.macro_terrain_indices, wgpu::BufferUsages::INDEX);
        let macro_o_vbuf = create_buf(&device, &queue, "macro_o_vbuf", &meshes.macro_ocean_verts, wgpu::BufferUsages::VERTEX);
        let macro_o_ibuf = create_buf(&device, &queue, "macro_o_ibuf", &meshes.macro_ocean_indices, wgpu::BufferUsages::INDEX);
        let sky_vbuf = create_buf(&device, &queue, "sky_vbuf", &meshes.sky_islands_verts, wgpu::BufferUsages::VERTEX);
        let sky_ibuf = create_buf(&device, &queue, "sky_ibuf", &meshes.sky_islands_indices, wgpu::BufferUsages::INDEX);

        let (hdr_view, depth_view, bg_resolve) = make_hdr_depth(&device, &resolve_bgl, 1920, 1080);

        let r_core = cfg.r_core as f32;
        let r_ocean = (cfg.r_core + cfg.z_sea as f64 + 0.55) as f32;
        let r_atmo = r_ocean + 216.0;

        Ok(Self {
            adapter_name,
            adapter_backend,
            device,
            queue,
            uni_near_buf,
            bg_uni_near,
            uni_macro_buf,
            bg_uni_macro,
            resolve_bgl,
            pipe_starfield,
            pipe_terrain,
            pipe_ocean,
            pipe_resolve,
            near_origin_f64: meshes.near_origin_f64,
            near_t_vbuf,
            near_t_ibuf,
            near_t_icount: meshes.near_terrain_indices.len() as u32,
            near_o_vbuf,
            near_o_ibuf,
            near_o_icount: meshes.near_ocean_indices.len() as u32,
            macro_t_vbuf,
            macro_t_ibuf,
            macro_t_icount: meshes.macro_terrain_indices.len() as u32,
            macro_o_vbuf,
            macro_o_ibuf,
            macro_o_icount: meshes.macro_ocean_indices.len() as u32,
            sky_vbuf,
            sky_ibuf,
            sky_icount: meshes.sky_islands_indices.len() as u32,
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

    fn render_into(&mut self, out_view: &wgpu::TextureView, w: u32, h: u32, shot: &CameraShotF64, time_s: f32) {
        self.ensure_targets(w, h);
        let (vp, inv_vp) = shot.view_proj_cam_rel(w, h);
        let cam_arr = [shot.pos.x, shot.pos.y, shot.pos.z];

        let near_off = world_f64_to_camera_rel_f32(self.near_origin_f64, cam_arr);
        let macro_off = world_f64_to_camera_rel_f32([0.0, 0.0, 0.0], cam_arr);

        let base_uni = |offset: [f32; 3]| HorizonUniforms {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: inv_vp.to_cols_array_2d(),
            cam_pos: [shot.pos.x as f32, shot.pos.y as f32, shot.pos.z as f32, time_s],
            model_offset: [offset[0], offset[1], offset[2], 1.0],
            sun_dir: [
                shot.sun_dir.x as f32,
                shot.sun_dir.y as f32,
                shot.sun_dir.z as f32,
                self.r_ocean,
            ],
            planet_cfg: [self.r_core, self.r_ocean - 0.5, self.r_atmo, shot.exposure],
            viewport: [w as f32, h as f32, 1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
        };

        let uni_near = base_uni(near_off);
        let uni_macro = base_uni(macro_off);
        self.queue.write_buffer(&self.uni_near_buf, 0, bytemuck::bytes_of(&uni_near));
        self.queue.write_buffer(&self.uni_macro_buf, 0, bytemuck::bytes_of(&uni_macro));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("horizon_enc"),
        });

        // Pass 1: Starfield + Layer 1 (Near 1m Shard + Hero Sky-Island) + Layer 2 (Global 1300m Macro Planet)
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("horizon_hdr_pass"),
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
            rp.set_bind_group(0, &self.bg_uni_macro, &[]);
            rp.draw(0..3, 0..1);

            // 1b. Near 1m Detail Shard + Pine Forests + Hero Sky-Island (floating-origin near_offset!)
            rp.set_pipeline(&self.pipe_terrain);
            rp.set_bind_group(0, &self.bg_uni_near, &[]);
            rp.set_vertex_buffer(0, self.near_t_vbuf.slice(..));
            rp.set_index_buffer(self.near_t_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.near_t_icount, 0, 0..1);

            // 1c. Global R=1300m Macro Planet Terrain + Sister Sky-Islands (floating-origin macro_offset!)
            rp.set_bind_group(0, &self.bg_uni_macro, &[]);
            rp.set_vertex_buffer(0, self.macro_t_vbuf.slice(..));
            rp.set_index_buffer(self.macro_t_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.macro_t_icount, 0, 0..1);

            if self.sky_icount > 0 {
                rp.set_vertex_buffer(0, self.sky_vbuf.slice(..));
                rp.set_index_buffer(self.sky_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.sky_icount, 0, 0..1);
            }

            // 1d. Near 1m Fresnel Ocean Bay
            rp.set_pipeline(&self.pipe_ocean);
            if self.near_o_icount > 0 {
                rp.set_bind_group(0, &self.bg_uni_near, &[]);
                rp.set_vertex_buffer(0, self.near_o_vbuf.slice(..));
                rp.set_index_buffer(self.near_o_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.near_o_icount, 0, 0..1);
            }

            // 1e. Global R=1300m Fresnel Sapphire Ocean Shell
            rp.set_bind_group(0, &self.bg_uni_macro, &[]);
            rp.set_vertex_buffer(0, self.macro_o_vbuf.slice(..));
            rp.set_index_buffer(self.macro_o_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.macro_o_icount, 0, 0..1);
        }

        // Pass 2: Seamless Ground-to-Orbit Rayleigh+Mie Atmosphere + ACES Tonemap
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("horizon_atmo_resolve_pass"),
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
            rp.set_bind_group(0, &self.bg_uni_macro, &[]);
            rp.set_bind_group(1, &self.bg_resolve, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(Some(enc.finish()));
    }

    fn create_out_tex(&self, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("horizon_out_rgba"),
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

    fn render_to_image(&mut self, w: u32, h: u32, shot: &CameraShotF64, time_s: f32) -> RgbaImage {
        let (tex, view) = self.create_out_tex(w, h);
        self.render_into(&view, w, h, shot, time_s);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let row_bytes = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("horizon_readback"),
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

    fn benchmark(&mut self, w: u32, h: u32, shot: &CameraShotF64, warmup: usize, iters: usize) -> ResBench {
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
// Crisp 5×7 Bitmap Badges for 2×2 Scale-Transition Composite
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
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
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
                p[0] = ((p[0] as u32 * 52 + 8 * 203) / 255) as u8;
                p[1] = ((p[1] as u32 * 52 + 12 * 203) / 255) as u8;
                p[2] = ((p[2] as u32 * 52 + 22 * 203) / 255) as u8;
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
// Metrics Schema
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
struct DualScaleMeshStats {
    near_1m_terrain_triangles: usize,
    near_1m_ocean_triangles: usize,
    macro_planet_terrain_triangles: usize,
    macro_planet_ocean_triangles: usize,
    sister_sky_islands_triangles: usize,
    total_triangles: usize,
}

#[derive(Serialize)]
struct PlanetHorizonMetrics {
    wave: &'static str,
    adapter: String,
    backend: String,
    r_core_m: f64,
    r_surface_m: f64,
    circumference_m: f64,
    horizon_drop_50m_m: f64,
    shards_per_face_axis: u8,
    total_shards: usize,
    shard_topology_directed_edges: usize,
    shard_size_1m: usize,
    face_size_virtual: usize,
    depth: usize,
    virtual_voxels: u64,
    global_climate_columns: usize,
    hydrated_hero_shard: PlanetShardKey,
    hydrated_1m_voxels: usize,
    hydrate_ms: f64,
    global_mass_error: i64,
    global_energy_error: i64,
    f64_floating_origin_err_1300m_m: f64,
    f64_floating_origin_err_earth_m: f64,
    mesh_stats: DualScaleMeshStats,
    bench_ground_1080p: ResBench,
    bench_ground_4k: ResBench,
    bench_orbital_1080p: ResBench,
    bench_orbital_4k: ResBench,
    showcase_outputs: Vec<String>,
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
    let showcase_dir = repo_root.join("models").join("planet_horizon_showcase");
    std::fs::create_dir_all(&showcase_dir)?;

    // 1. Construct live Dual-Scale Locally-Flat Planet (`R_core = 1280m, R_surf ≈ 1304m`,
    //    `6 × 2048² × 64 = 1.61B` virtual voxels, `8×8 = 64` shards/face = `384` shards!)
    let cfg = MacroPlanetConfig::default();
    let hero_sun = normalize3_f64([-0.72, 0.36, 0.59]);
    let r_surf = cfg.surface_radius_m();
    let drop_50m = horizon_drop_meters(50.0, r_surf);

    // Verify all 1,536 directed shard topology edges satisfy exact involution at runtime!
    let mut verified_edges = 0usize;
    for f in 0..FACE_COUNT as u8 {
        for sy in 0..cfg.shards_per_face_axis {
            for sx in 0..cfg.shards_per_face_axis {
                let k = PlanetShardKey::new(f, sx, sy);
                for e in 0..EDGE_COUNT {
                    let ab = shard_neighbor(k, e, cfg.shards_per_face_axis);
                    let ba = shard_neighbor(ab.neighbor, ab.edge, cfg.shards_per_face_axis);
                    assert_eq!(ba.neighbor, k);
                    assert_eq!(ba.edge, e);
                    assert_eq!(ab.det(), 1);
                    verified_edges += 1;
                }
            }
        }
    }

    println!(
        "[planet_horizon_demo] DualScalePlanet R_surf={:.1}m | circ={:.1}m | drop_50m={:.4}m | virtual_voxels={} | shard_edges={verified_edges}",
        r_surf,
        cfg.circumference_m(),
        drop_50m,
        cfg.virtual_voxels()
    );

    let mut planet = DualScalePlanet::new(cfg);
    planet.sun_dir = hero_sun;

    // 2. Locate a sunlit scenic golden-beach / sapphire-bay / pine-hill hero shard and place our
    //    Hero Floating Sky-Island right over the bay BEFORE hydrating the `256×256×64` `1 m` shard!
    let site = find_scenic_coastal_hero_site(cfg, hero_sun);
    planet.sky_islands[0] = MacroSkyIsland {
        id: 0,
        name: "AETHER-CITADEL-PRIME".to_string(),
        center_dir: site.island_dir,
        radius_m: 28.0,
        z_bot: 46.0,
        z_top: 57.5,
        core_radius_m: 9.2,
        phase: 0.85,
        has_waterfall: true,
    };

    let t_hyd = Instant::now();
    planet.hydrate_shard(site.shard_key);
    let hydrate_ms = t_hyd.elapsed().as_secs_f64() * 1000.0;

    // Run bidirectional macro <-> micro climate coupling on the live hydrated 256×256×64 shard!
    planet.apply_climate_to_shard(site.shard_key);
    planet.upscale_shard_to_climate(site.shard_key);
    planet.climate.step_climate(hero_sun, 0.1);
    let mass_err = planet.global_mass_error();
    let energy_err = planet.global_energy_error();
    assert_eq!(mass_err, 0, "global_mass_error MUST equal 0");
    assert_eq!(energy_err, 0, "global_energy_error MUST equal 0");

    let hydrated_voxels = cfg.shard_size * cfg.shard_size * cfg.depth;
    println!(
        "[planet_horizon_demo] Hydrated 1m shard {:?} ({hydrated_voxels} voxels) in {:.1} ms | mass_err={mass_err} energy_err={energy_err}",
        site.shard_key, hydrate_ms
    );

    // Measure exact `f64` floating-origin precision at R = 1,300 m and Earth R = 6,371,000 m
    let measure_fo_err = |radius_m: f64| -> f64 {
        let cam = face_uv_z_to_world_f64(4, 0.0, 0.0, 2.5, radius_m);
        let pt = [cam[0] + 12.3456, cam[1] - 7.8901, cam[2] + 4.5678];
        let rel = world_f64_to_camera_rel_f32(pt, cam);
        ((rel[0] as f64 - 12.3456).powi(2)
            + (rel[1] as f64 + 7.8901).powi(2)
            + (rel[2] as f64 - 4.5678).powi(2))
        .sqrt()
    };
    let fo_err_1300m = measure_fo_err(r_surf);
    let fo_err_earth = measure_fo_err(6_371_000.0);

    // 3. Extract synchronized Dual-Scale curved meshes (Layer 1: 1m shard + Layer 2: Global 1300m shell)
    let t_mesh = Instant::now();
    let meshes = build_dual_scale_meshes(&planet, site.shard_key, site.waterfall_az);
    let mesh_stats = DualScaleMeshStats {
        near_1m_terrain_triangles: meshes.near_terrain_indices.len() / 3,
        near_1m_ocean_triangles: meshes.near_ocean_indices.len() / 3,
        macro_planet_terrain_triangles: meshes.macro_terrain_indices.len() / 3,
        macro_planet_ocean_triangles: meshes.macro_ocean_indices.len() / 3,
        sister_sky_islands_triangles: meshes.sky_islands_indices.len() / 3,
        total_triangles: (meshes.near_terrain_indices.len()
            + meshes.near_ocean_indices.len()
            + meshes.macro_terrain_indices.len()
            + meshes.macro_ocean_indices.len()
            + meshes.sky_islands_indices.len())
            / 3,
    };
    println!(
        "[planet_horizon_demo] Dual-Scale meshes built in {:.1} ms: {} total tris ({} near 1m + {} macro globe)",
        t_mesh.elapsed().as_secs_f64() * 1000.0,
        mesh_stats.total_triangles,
        mesh_stats.near_1m_terrain_triangles + mesh_stats.near_1m_ocean_triangles,
        mesh_stats.macro_planet_terrain_triangles + mesh_stats.macro_planet_ocean_triangles
    );

    // 4. Initialize Apple M4 Pro Metal renderer
    let rt = tokio::runtime::Builder::new_current_thread().build()?;
    let mut renderer = rt.block_on(MetalHorizonRenderer::new(&meshes, cfg))?;
    println!(
        "[planet_horizon_demo] GPU: {} ({})",
        renderer.adapter_name, renderer.adapter_backend
    );

    // 5. Define the 4 iconic Seamless Scale-Transition Camera Shots (all in exact `f64`!)
    let b_dir = DVec3::from_array(site.beach_dir).normalize();
    let b_fwd = DVec3::from_array(site.look_fwd).normalize();
    let b_right = b_fwd.cross(b_dir).normalize();
    let isl_dir = DVec3::from_array(site.island_dir).normalize();
    let sun_v = DVec3::from_array(hero_sun).normalize();

    // Shot 1: Ground Eye-Level (`2.5 m` above 1m golden beach/meadow) — 100% locally flat vista!
    let r_ground_eye = cfg.r_core + site.beach_surf_z.max(cfg.z_sea as f64 + 1.1) + 2.5;
    let ground_pos = b_dir * r_ground_eye;
    let ground_look_dir = (b_fwd + b_dir * 0.082 + b_right * 0.06).normalize();
    let shot_ground = CameraShotF64 {
        pos: ground_pos,
        target: ground_pos + ground_look_dir * 120.0,
        up: b_dir,
        fov_y_deg: 58.0,
        near: 0.25,
        far: 2600.0,
        sun_dir: sun_v,
        exposure: 1.04,
    };

    // Shot 2: Skywing Flight (`120 m` altitude) soaring past the glowing Sky-Island looking across
    //         the pine-forested coastline & gently curving sapphire planetary horizon!
    let flight_dir = (b_dir - b_fwd * (22.0 / r_surf) - b_right * (76.0 / r_surf)).normalize();
    let flight_pos = flight_dir * (r_surf + 120.0);
    let isl_world = isl_dir * (cfg.r_core + 44.0);
    let flight_to_isl = (isl_world - flight_pos).normalize();
    let flight_fwd = (flight_to_isl * 0.80 + b_fwd * 0.28 + flight_dir * 0.06).normalize();
    let shot_low_flight = CameraShotF64 {
        pos: flight_pos,
        target: flight_pos + flight_fwd * 320.0,
        up: flight_dir,
        fov_y_deg: 58.0,
        near: 0.8,
        far: 4200.0,
        sun_dir: sun_v,
        exposure: 1.06,
    };

    // Shot 3: Stratosphere (`600 m` altitude) — coastline & continents bending into the glowing blue limb!
    let strat_dir = (b_dir - b_fwd * (260.0 / r_surf) - b_right * (140.0 / r_surf)).normalize();
    let strat_pos = strat_dir * (r_surf + 600.0);
    let strat_target = (b_dir + b_fwd * (420.0 / r_surf)).normalize() * r_surf;
    let shot_stratosphere = CameraShotF64 {
        pos: strat_pos,
        target: strat_target,
        up: strat_dir,
        fov_y_deg: 60.0,
        near: 4.0,
        far: 6500.0,
        sun_dir: sun_v,
        exposure: 1.12,
    };

    // Shot 4: Full Orbital Globe (`3,300 m` from center) — complete closed R=1,300m planet in space
    //         with Day/Night Terminator, continents, Fresnel oceans, polar cap, glowing volcanic rift & starfield!
    let orbital_pos = DVec3::new(0.56, 0.20, 0.805).normalize() * 3300.0;
    let shot_orbital = CameraShotF64 {
        pos: orbital_pos,
        target: DVec3::ZERO,
        up: DVec3::Y,
        fov_y_deg: 56.0,
        near: 30.0,
        far: 10000.0,
        sun_dir: sun_v,
        exposure: 1.24,
    };

    // 6. Render & save the 3 full 1080p hero PNGs + the 2×2 Scale-Transition Showcase!
    let p_ground = showcase_dir.join("horizon_ground_1080p.png");
    let p_flight = showcase_dir.join("horizon_low_flight_1080p.png");
    let p_orbital = showcase_dir.join("horizon_orbital_1080p.png");
    let p_4view = showcase_dir.join("horizon_4view.png");

    let img_ground = renderer.render_to_image(1920, 1080, &shot_ground, 1.0);
    img_ground.save(&p_ground)?;
    println!("[planet_horizon_demo] Saved {}", p_ground.display());

    let img_flight = renderer.render_to_image(1920, 1080, &shot_low_flight, 1.0);
    img_flight.save(&p_flight)?;
    println!("[planet_horizon_demo] Saved {}", p_flight.display());

    let img_orbital = renderer.render_to_image(1920, 1080, &shot_orbital, 1.0);
    img_orbital.save(&p_orbital)?;
    println!("[planet_horizon_demo] Saved {}", p_orbital.display());

    // Render crisp 960×540 quadrants for the 1920×1080 2×2 Seamless Scale-Transition Showcase
    let q_tl = renderer.render_to_image(960, 540, &shot_ground, 1.0);
    let q_tr = renderer.render_to_image(960, 540, &shot_low_flight, 1.0);
    let q_bl = renderer.render_to_image(960, 540, &shot_stratosphere, 1.0);
    let q_br = renderer.render_to_image(960, 540, &shot_orbital, 1.0);

    let mut composite = RgbaImage::new(1920, 1080);
    blit_quadrant(&mut composite, &q_tl, 0, 0);
    blit_quadrant(&mut composite, &q_tr, 960, 0);
    blit_quadrant(&mut composite, &q_bl, 0, 540);
    blit_quadrant(&mut composite, &q_br, 960, 540);

    for x in 0..1920 {
        composite.put_pixel(x, 539, Rgba([30, 46, 74, 255]));
        composite.put_pixel(x, 540, Rgba([30, 46, 74, 255]));
    }
    for y in 0..1080 {
        composite.put_pixel(959, y, Rgba([30, 46, 74, 255]));
        composite.put_pixel(960, y, Rgba([30, 46, 74, 255]));
    }

    draw_badge(&mut composite, 16, 16, "1. GROUND 2.5M: LOCALLY FLAT 1M SHARD", [72, 224, 152]);
    draw_badge(&mut composite, 976, 16, "2. SKYWING 120M: SKY-ISLAND & COAST", [86, 212, 255]);
    draw_badge(&mut composite, 16, 556, "3. STRATOSPHERE 600M: CURVING LIMB", [122, 168, 255]);
    draw_badge(&mut composite, 976, 556, "4. ORBIT 3,300M: R=1,300M 1.61B PLANET", [255, 164, 78]);
    composite.save(&p_4view)?;
    println!("[planet_horizon_demo] Saved {}", p_4view.display());

    // 7. Benchmark 1080p & 4K Metal FPS (Ground Dual-Scale + Orbital Globe)
    let bench_ground_1080p = renderer.benchmark(1920, 1080, &shot_ground, 5, 25);
    let bench_ground_4k = renderer.benchmark(3840, 2160, &shot_ground, 5, 25);
    let bench_orbital_1080p = renderer.benchmark(1920, 1080, &shot_orbital, 5, 25);
    let bench_orbital_4k = renderer.benchmark(3840, 2160, &shot_orbital, 5, 25);
    println!(
        "[planet_horizon_demo] Ground 1080p: {:.3}ms ({:.1} FPS) | Ground 4K: {:.3}ms ({:.1} FPS)",
        bench_ground_1080p.p50_ms,
        bench_ground_1080p.fps_p50,
        bench_ground_4k.p50_ms,
        bench_ground_4k.fps_p50
    );
    println!(
        "[planet_horizon_demo] Orbital 1080p: {:.3}ms ({:.1} FPS) | Orbital 4K: {:.3}ms ({:.1} FPS)",
        bench_orbital_1080p.p50_ms,
        bench_orbital_1080p.fps_p50,
        bench_orbital_4k.p50_ms,
        bench_orbital_4k.fps_p50
    );

    // 8. Save `models/PLANET_HORIZON_METRICS.json`
    let metrics = PlanetHorizonMetrics {
        wave: "Wave 33B: Dual-Scale Locally-Flat R=1,300m Cubed-Sphere Planet (1.61B Virtual Voxels) + Apple M4 Pro Metal Proof",
        adapter: renderer.adapter_name,
        backend: renderer.adapter_backend,
        r_core_m: cfg.r_core,
        r_surface_m: r_surf,
        circumference_m: round2(cfg.circumference_m()),
        horizon_drop_50m_m: (drop_50m * 10000.0).round() / 10000.0,
        shards_per_face_axis: cfg.shards_per_face_axis,
        total_shards: cfg.total_shards(),
        shard_topology_directed_edges: verified_edges,
        shard_size_1m: cfg.shard_size,
        face_size_virtual: cfg.face_size_virtual(),
        depth: cfg.depth,
        virtual_voxels: cfg.virtual_voxels(),
        global_climate_columns: planet.climate.total_columns(),
        hydrated_hero_shard: site.shard_key,
        hydrated_1m_voxels: hydrated_voxels,
        hydrate_ms: round2(hydrate_ms),
        global_mass_error: mass_err,
        global_energy_error: energy_err,
        f64_floating_origin_err_1300m_m: fo_err_1300m,
        f64_floating_origin_err_earth_m: fo_err_earth,
        mesh_stats,
        bench_ground_1080p,
        bench_ground_4k,
        bench_orbital_1080p,
        bench_orbital_4k,
        showcase_outputs: vec![
            "models/planet_horizon_showcase/horizon_ground_1080p.png".to_string(),
            "models/planet_horizon_showcase/horizon_low_flight_1080p.png".to_string(),
            "models/planet_horizon_showcase/horizon_orbital_1080p.png".to_string(),
            "models/planet_horizon_showcase/horizon_4view.png".to_string(),
        ],
    };

    let metrics_path = repo_root.join("models").join("PLANET_HORIZON_METRICS.json");
    std::fs::write(&metrics_path, serde_json::to_string_pretty(&metrics)?)?;
    println!("[planet_horizon_demo] Saved {}", metrics_path.display());

    Ok(())
}
