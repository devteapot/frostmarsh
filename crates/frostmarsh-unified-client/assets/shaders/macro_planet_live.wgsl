// Wave 34B: Live Locally-Flat R = 1,304 m Dual-Scale Spherical Clipmap +
//           Live Coriolis PlanetaryClimateNet Clouds/Storms Metal Shader (`macro_planet_live.wgsl`)
//
// Pipelines:
//   1. vs_fullscreen   + fs_starfield               -> Deep cosmic starfield, nebula band & live Sun disc
//   2. vs_macro_planet + fs_macro_solid             -> Floating-origin near 1m shards, Layer A 6-face R=1304m
//                                                      macro globe, 3D Nordic pines, Floating Sky-Islands +
//                                                      cyan-amethyst Buoyant Crystal Cores, 60 live 3D heroes,
//                                                      shrines & solid projectiles (with live snowline & cloud shadows!)
//   3. vs_macro_planet + fs_macro_ocean             -> Curved Fresnel sapphire ocean with surf foam & live sea-ice
//   4. vs_macro_planet + fs_macro_translucent       -> Geodesic actuator beams, shrine light pillars, aura rings
//                                                      & translucent steam / volcanic ash plumes
//   5. vs_fullscreen   + fs_macro_atmosphere_resolve-> Seamless Ground (2.5m) -> Flight (120m) -> Stratosphere (600m)
//                                                      -> Orbit (3,300m) Rayleigh + Mie atmosphere + Live 6-face
//                                                      PlanetaryClimateNet Coriolis Cloud & Ash Deck + ACES Filmic

struct MacroPlanetUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,        // xyz = planet-centered camera pos (m), w = sim_time
    model_offset: vec4<f32>,   // xyz = (mesh_origin_f64 - cam_pos_f64) as f32, w = r_ocean
    sun_dir: vec4<f32>,        // xyz = normalized sun dir, w = r_cloud_shell
    planet_cfg: vec4<f32>,     // x = r_core (1280), y = r_surf_atmo (1304), z = r_atmo_outer (1520), w = exposure
    viewport: vec4<f32>,       // x = width, y = height, z = 1/width, w = 1/height
    climate_params: vec4<f32>, // x = local_wind_u, y = local_wind_v, z = local_cloud_cover, w = has_live_climate
};

@group(0) @binding(0) var<uniform> uni: MacroPlanetUniforms;
@group(0) @binding(1) var climate_tex: texture_2d_array<f32>;
@group(0) @binding(2) var climate_samp: sampler;

// -----------------------------------------------------------------------------
// Seamless 3D Spherical Procedural Noise & Exact 6-Face Cubed-Sphere Climate Lookup
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

/// Exact inverse cubed-sphere projection matching `field_core::planet::FACE_FRAMES`.
/// Returns RGBA8 live `PlanetaryClimateNet`:
///   R = temp_norm (240..320K), G = cloud_storm, B = ash_plume, A = ice_snow.
fn sample_climate_dir(dir_in: vec3<f32>) -> vec4<f32> {
    let dir = normalize(dir_in);
    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);

    var face_idx: i32 = 4;
    var u: f32 = 0.0;
    var v: f32 = 0.0;

    if (ax >= ay && ax >= az) {
        if (dir.x >= 0.0) {
            face_idx = 0; // +X: e_u=(0,0,-1), e_v=(0,1,0)
            let inv = 1.0 / max(dir.x, 1e-5);
            u = -dir.z * inv;
            v = dir.y * inv;
        } else {
            face_idx = 1; // -X: e_u=(0,0,1), e_v=(0,1,0)
            let inv = 1.0 / max(-dir.x, 1e-5);
            u = dir.z * inv;
            v = dir.y * inv;
        }
    } else if (ay >= ax && ay >= az) {
        if (dir.y >= 0.0) {
            face_idx = 2; // +Y: e_u=(1,0,0), e_v=(0,0,-1)
            let inv = 1.0 / max(dir.y, 1e-5);
            u = dir.x * inv;
            v = -dir.z * inv;
        } else {
            face_idx = 3; // -Y: e_u=(1,0,0), e_v=(0,0,1)
            let inv = 1.0 / max(-dir.y, 1e-5);
            u = dir.x * inv;
            v = dir.z * inv;
        }
    } else {
        if (dir.z >= 0.0) {
            face_idx = 4; // +Z: e_u=(1,0,0), e_v=(0,1,0)
            let inv = 1.0 / max(dir.z, 1e-5);
            u = dir.x * inv;
            v = dir.y * inv;
        } else {
            face_idx = 5; // -Z: e_u=(-1,0,0), e_v=(0,1,0)
            let inv = 1.0 / max(-dir.z, 1e-5);
            u = -dir.x * inv;
            v = dir.y * inv;
        }
    }

    let uv = clamp(vec2<f32>(u, v) * 0.5 + 0.5, vec2<f32>(0.004), vec2<f32>(0.996));
    return textureSampleLevel(climate_tex, climate_samp, uv, face_idx, 0.0);
}

// -----------------------------------------------------------------------------
// Pass 1A: Full-Screen Cosmic Starfield + Galactic Nebula + Live Sun Disc
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
    let t = uni.cam_pos.w;

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
        let tw = 0.88 + 0.12 * sin(t * (1.8 + fract(h1 * 37.0) * 2.4) + h1 * 80.0);
        let b = smoothstep(0.32, 0.04, d) * (0.55 + 1.45 * fract(h1 * 137.0)) * tw;
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
// Floating-Origin Dual-Scale Vertex I/O
//   color:        rgb = albedo/tint, a = roughness (solid) or alpha (translucent)
//   emissive_mat: rgb = RGB emissive radiance, w = mat_kind
//     0.0 = Terrain (near 1m shard + macro globe)
//     1.0 = Prop (Nordic pines, boulders, sculpted Sky-Islands, waterfalls)
//     2.0 = Fresnel Sapphire Ocean
//     3.0 = Pulsing Cyan-Amethyst Buoyant Crystal Core
//     4.0 = Live 3D Hero / Creature / Shrine Obelisk
//     5.0 = Geodesic Beam / Projectile / Shrine Light Pillar / Aura Ring
//     6.0 = Translucent Steam / Volcanic Ash Plume
//   water_ice:    x = water_depth, y = ice_frac, z = is_macro, w = reserved
// -----------------------------------------------------------------------------
struct VertIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) emissive_mat: vec4<f32>,
    @location(4) water_ice: vec4<f32>,
};

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) rel_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) emissive_mat: vec4<f32>,
    @location(5) water_ice: vec4<f32>,
};

@vertex
fn vs_macro_planet(in: VertIn) -> VertOut {
    var out: VertOut;
    // Exact camera-relative floating-origin position (camera sits strictly at Vec3::ZERO!)
    let rel_pos = in.pos + uni.model_offset.xyz;
    out.clip_pos = uni.view_proj * vec4<f32>(rel_pos, 1.0);
    out.rel_pos = rel_pos;
    out.world_pos = rel_pos + uni.cam_pos.xyz;
    out.normal = in.normal;
    out.color = in.color;
    out.emissive_mat = in.emissive_mat;
    out.water_ice = in.water_ice;
    return out;
}

// -----------------------------------------------------------------------------
// Pass 1B: Dual-Scale Terrain + Pines + Sky-Islands + Crystal Cores + 60 Heroes/Shrines
// -----------------------------------------------------------------------------
@fragment
fn fs_macro_solid(in: VertOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(-in.rel_pos);
    let dist_cam = length(in.rel_pos);
    let t = uni.cam_pos.w;

    let mat_kind = in.emissive_mat.w;
    let em_rgb = max(in.emissive_mat.rgb, vec3<f32>(0.0));
    let rough = clamp(in.color.a, 0.08, 0.96);

    let term_raw = dot(radial, sun);
    let day_mask = smoothstep(-0.14, 0.22, term_raw);
    let twilight = exp(-pow((term_raw - 0.02) * 5.2, 2.0));
    let sun_col = mix(
        vec3<f32>(1.0, 0.48, 0.18),
        vec3<f32>(1.08, 1.00, 0.90),
        smoothstep(-0.04, 0.30, term_raw),
    );

    // 1. Luminous projectiles / beam cores (mat_kind >= 4.5)
    if (mat_kind > 4.5) {
        let pulse = 0.90 + 0.16 * sin(t * 7.2 + length(in.rel_pos) * 0.25);
        let ndv = max(dot(normalize(in.normal), view_dir), 0.0);
        let core = mix(0.75, 1.45, pow(ndv, 1.3));
        let lit = in.color.rgb * 0.35 + em_rgb * pulse * core;
        return vec4<f32>(lit, 1.0);
    }

    // 2. Live 3D Heroes, Creatures & Planetary Shrine Obelisks (mat_kind ~ 4.0)
    if (mat_kind > 3.5) {
        let n = normalize(in.normal);
        let ndl = max(dot(n, sun), 0.0);
        let wrap_ndl = max((dot(n, sun) + 0.28) / 1.28, 0.0);
        let sky_up = 0.5 + 0.5 * dot(n, radial);
        let amb = mix(vec3<f32>(0.09, 0.13, 0.22), vec3<f32>(0.26, 0.38, 0.56), sky_up)
            * mix(0.60, 1.0, day_mask);
        let diff = sun_col * (ndl * 1.48 * day_mask + wrap_ndl * 0.42 * twilight + 0.22 * (1.0 - day_mask));

        let half_v = normalize(sun + view_dir);
        let shininess = mix(72.0, 12.0, rough);
        let spec = pow(max(dot(n, half_v), 0.0), shininess) * mix(0.55, 0.12, rough) * (0.35 + 0.65 * day_mask);

        let fres = pow(1.0 - max(dot(n, view_dir), 0.0), 2.4);
        let rim = in.color.rgb * fres * mix(0.75, 0.30, day_mask);
        let pulse = 0.92 + 0.12 * sin(t * 4.5 + p.x * 0.25 + p.z * 0.25);
        let emissive = em_rgb * pulse * mix(1.45, 1.05, day_mask);

        let lit = in.color.rgb * (diff + amb) + sun_col * spec + rim + emissive;
        return vec4<f32>(lit, 1.0);
    }

    // 3. Radiant Cyan-Amethyst Buoyant Crystal Core (mat_kind ~ 3.0)
    if (mat_kind > 2.5) {
        let n = normalize(in.normal);
        let pulse = 0.86 + 0.22 * sin(t * 3.6 + dot(p, vec3<f32>(0.15, 0.12, 0.18)));
        let ndv = max(dot(n, view_dir), 0.0);
        let fres = pow(1.0 - ndv, 2.0);
        let crystal = mix(vec3<f32>(0.12, 0.82, 1.0), vec3<f32>(0.72, 0.28, 1.0), fres * 0.72);
        let emissive = max(em_rgb, crystal * 1.65) * pulse * mix(1.30, 0.92, day_mask);
        let lit = in.color.rgb * (0.22 + 0.45 * day_mask) + emissive;
        return vec4<f32>(lit, 1.0);
    }

    // 4. Dual-Scale Terrain & 3D Props (mat_kind 0.0 = terrain, 1.0 = pines/boulders/sky-islands/waterfalls)
    let blend_far = smoothstep(140.0, 950.0, dist_cam);
    let f_cur = mix(0.38, 0.045, blend_far);

    let micro = fbm3(p * f_cur);
    let micro_fine = noise3(p * (f_cur * 4.2));
    let is_prop = step(0.5, mat_kind);
    let is_foliage = step(0.84, rough) * step(in.color.g, 0.36) * step(in.color.r, 0.18);
    let detail_mod = mix(0.80 + 0.32 * micro + 0.10 * (micro_fine - 0.5), 0.88 + 0.24 * micro_fine, is_foliage);

    var n = normalize(in.normal);
    let eps = mix(0.28, 2.2, blend_far);
    let gx = fbm3((p + vec3<f32>(eps, 0.0, 0.0)) * f_cur) - fbm3((p - vec3<f32>(eps, 0.0, 0.0)) * f_cur);
    let gy = fbm3((p + vec3<f32>(0.0, eps, 0.0)) * f_cur) - fbm3((p - vec3<f32>(0.0, eps, 0.0)) * f_cur);
    let gz = fbm3((p + vec3<f32>(0.0, 0.0, eps)) * f_cur) - fbm3((p - vec3<f32>(0.0, 0.0, eps)) * f_cur);
    let grad = vec3<f32>(gx, gy, gz);
    let bump_strength = mix(0.72, 0.15, max(is_foliage, is_prop * 0.5)) * (1.0 - 0.35 * blend_far);
    n = normalize(n - (grad - radial * dot(grad, radial)) * bump_strength);

    let slope = clamp(1.0 - max(dot(n, radial), 0.0), 0.0, 1.0);
    let rock_col = vec3<f32>(0.31, 0.32, 0.36) * (0.85 + 0.30 * micro_fine);
    let is_emissive = step(0.12, length(em_rgb));
    let rock_blend = smoothstep(0.24, 0.52, slope) * (1.0 - is_emissive) * (1.0 - is_prop) * 0.68;
    var base_albedo = mix(in.color.rgb, rock_col, rock_blend);

    // Live PlanetaryClimateNet dynamic snowline (A), volcanic ash dust (B), & soft storm cloud shadows (G)!
    let clim = sample_climate_dir(radial);
    let has_clim = uni.climate_params.w;
    let snow_frost = smoothstep(0.46, 0.84, clim.a) * (1.0 - is_emissive) * (1.0 - is_prop * 0.75) * (1.0 - smoothstep(0.32, 0.65, slope)) * has_clim;
    base_albedo = mix(base_albedo, vec3<f32>(0.84, 0.90, 0.96), snow_frost * 0.68);
    let ash_dust = smoothstep(0.35, 0.85, clim.b) * (1.0 - is_emissive) * (1.0 - is_prop) * has_clim;
    base_albedo = mix(base_albedo, vec3<f32>(0.16, 0.15, 0.17), ash_dust * 0.42);
    let cloud_shadow = 1.0 - smoothstep(0.34, 0.88, clim.g) * 0.34 * has_clim;

    let ndl = max(dot(n, sun), 0.0);
    let wrap_ndl = max((dot(n, sun) + 0.16) / 1.16, 0.0);
    let diffuse = sun_col * (ndl * 1.52 * day_mask * cloud_shadow + wrap_ndl * 0.30 * twilight);

    let sky_up = 0.5 + 0.5 * dot(n, radial);
    let day_amb = mix(vec3<f32>(0.05, 0.08, 0.13), vec3<f32>(0.16, 0.29, 0.48), sky_up) * (0.18 + 0.82 * day_mask);
    let night_amb = vec3<f32>(0.009, 0.015, 0.030) * (1.0 - day_mask);

    let half_v = normalize(sun + view_dir);
    let eff_rough = mix(rough, 0.26, snow_frost * 0.65);
    let shininess = mix(96.0, 6.0, eff_rough);
    let spec = pow(max(dot(n, half_v), 0.0), shininess) * mix(0.46, 0.04, eff_rough) * day_mask * cloud_shadow;

    let fres = pow(1.0 - max(dot(n, view_dir), 0.0), 3.4);
    let rim_col = mix(vec3<f32>(0.12, 0.36, 0.78), vec3<f32>(0.95, 0.44, 0.16), twilight)
        * fres
        * (0.16 * day_mask + 0.24 * twilight);

    let albedo = base_albedo * detail_mod;
    let em_pulse = 0.86 + 0.30 * smoothstep(0.25, 0.85, micro + 0.05 * sin(t * 2.0));
    let emissive = em_rgb * em_pulse * mix(1.65, 1.05, day_mask);

    let lit = albedo * (diffuse + day_amb + night_amb) + sun_col * spec + rim_col + emissive;
    return vec4<f32>(lit, 1.0);
}

// -----------------------------------------------------------------------------
// Pass 1C: Curved Specular/Fresnel Sapphire Ocean & Coastal Turquoise Shallows
// -----------------------------------------------------------------------------
@fragment
fn fs_macro_ocean(in: VertOut) -> @location(0) vec4<f32> {
    let water_depth = in.water_ice.x;
    if (water_depth < 0.04) {
        discard;
    }
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(-in.rel_pos);
    let dist_cam = length(in.rel_pos);
    let blend_far = smoothstep(140.0, 950.0, dist_cam);

    let clim = sample_climate_dir(radial);
    let has_clim = uni.climate_params.w;
    let ice_frac = clamp(max(in.water_ice.y, smoothstep(0.76, 0.96, clim.a) * 0.72 * has_clim), 0.0, 1.0);
    let cloud_shadow = 1.0 - smoothstep(0.36, 0.88, clim.g) * 0.30 * has_clim;
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

    let shallow_col = vec3<f32>(0.04, 0.52, 0.66);
    let mid_col = vec3<f32>(0.015, 0.17, 0.45);
    let deep_col = vec3<f32>(0.006, 0.048, 0.20);
    let d_norm = clamp(water_depth / 4.2, 0.0, 1.0);
    var water_col = mix(shallow_col, mid_col, smoothstep(0.0, 0.36, d_norm));
    water_col = mix(water_col, deep_col, smoothstep(0.32, 1.0, d_norm));

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
    let spec = sun_col * (spec_sharp + spec_broad) * (day_mask * cloud_shadow + twilight * 0.45);

    let ndl = max(dot(n, sun), 0.0);
    let diff = base_col * (
        sun_col * ndl * 1.28 * day_mask * cloud_shadow
            + vec3<f32>(0.09, 0.18, 0.34) * (0.20 + 0.80 * day_mask)
            + vec3<f32>(0.008, 0.014, 0.032) * (1.0 - day_mask)
    );

    let col = mix(diff, sky_refl, fresnel * mix(0.68, 0.22, ice_frac)) + spec;
    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Pass 1D: Translucent Beams, Shrine Light Pillars, Aura Rings & Steam/Ash Plumes
// -----------------------------------------------------------------------------
@fragment
fn fs_macro_translucent(in: VertOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let t = uni.cam_pos.w;
    let mat_kind = in.emissive_mat.w;
    let em_rgb = max(in.emissive_mat.rgb, vec3<f32>(0.0));
    let alpha = clamp(in.color.a, 0.0, 1.0);

    // Volumetric steam / volcanic ash plumes (mat_kind >= 5.5)
    if (mat_kind > 5.5) {
        let puff = fbm3(p * 0.18 - radial * (t * 0.45));
        let term_raw = dot(radial, sun);
        let day_mask = smoothstep(-0.18, 0.24, term_raw);
        let twilight = exp(-pow((term_raw - 0.02) * 4.5, 2.0));
        let sun_tint = mix(vec3<f32>(0.24, 0.30, 0.45), vec3<f32>(1.05, 0.98, 0.92), day_mask)
            + vec3<f32>(1.0, 0.48, 0.18) * twilight * 0.55;
        let col = in.color.rgb * sun_tint * (0.75 + 0.45 * puff) + em_rgb;
        let a = alpha * smoothstep(0.18, 0.80, puff);
        if (a < 0.02) {
            discard;
        }
        return vec4<f32>(col, a);
    }

    // Glowing Shrine Light Pillars, Hero Aura Rings & Curved Geodesic Actuator Beams
    let pulse = 0.88 + 0.18 * sin(t * 5.2 + length(in.rel_pos) * 0.18);
    let col = in.color.rgb * 0.55 + em_rgb * pulse;
    return vec4<f32>(col, alpha);
}

// -----------------------------------------------------------------------------
// Pass 2: Seamless Ground-to-Orbit Rayleigh + Mie Atmosphere +
//         Live PlanetaryClimateNet Coriolis Cloud & Volcanic Ash Deck + ACES
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

/// Evaluate live `PlanetaryClimateNet` Coriolis storm clouds & volcanic ash plumes at sphere dir `c_rad`.
fn eval_coriolis_cloud_layer(c_rad: vec3<f32>, sun: vec3<f32>, t: f32, is_above: f32) -> vec4<f32> {
    // Gentle zonal trade-wind / Coriolis swirl drift
    let lat = c_rad.y;
    let drift_ang = t * 0.008 * (1.0 - 0.65 * abs(lat));
    let cs = cos(drift_ang);
    let sn = sin(drift_ang);
    let drifted = vec3<f32>(c_rad.x * cs - c_rad.z * sn, c_rad.y, c_rad.x * sn + c_rad.z * cs);

    let clim = sample_climate_dir(c_rad);
    let live_cloud = clim.g;
    let live_ash = clim.b;

    // Multi-scale swirling Coriolis filament noise modulating the 64x64 live climate field
    let swirl = vec3<f32>(
        sin(lat * 8.0 + c_rad.z * 3.5) * 0.18,
        0.0,
        cos(lat * 8.0 - c_rad.x * 3.5) * 0.18,
    );
    let n_macro = fbm3((drifted + swirl) * 7.5 + vec3<f32>(1.4, -0.6, 2.1));
    let n_fine = fbm3(drifted * 24.0 + vec3<f32>(-2.2, 1.3, 0.9));
    let billow = n_macro * 0.64 + n_fine * 0.36;

    // Combine live PlanetaryClimateNet storm field with procedural fine wisp detail
    let cloud_raw = live_cloud * (0.62 + 0.72 * billow) + smoothstep(0.62, 0.88, billow) * 0.16;
    let cloud_cov = smoothstep(0.24, 0.80, cloud_raw);
    let ash_cov = smoothstep(0.22, 0.78, live_ash * (0.68 + 0.65 * n_fine));

    // Sunlit cloud deck shading across the Day/Night Terminator
    let term = dot(c_rad, sun);
    let day_m = smoothstep(-0.14, 0.24, term);
    let twi_m = exp(-pow((term - 0.02) * 5.0, 2.0));
    let cloud_day = mix(vec3<f32>(0.76, 0.84, 0.94), vec3<f32>(1.04, 1.01, 0.97), smoothstep(0.30, 0.82, billow));
    let cloud_twi = vec3<f32>(1.05, 0.54, 0.26) * twi_m * 0.72;
    let cloud_night = vec3<f32>(0.022, 0.032, 0.058) * (1.0 - day_m);
    let cloud_col = cloud_day * (0.14 + 0.92 * day_m) + cloud_twi + cloud_night;

    // Volcanic ash plume shading (smoky dark umber with warm caldera underglow)
    let ash_col = mix(vec3<f32>(0.08, 0.07, 0.09), vec3<f32>(0.34, 0.26, 0.22), day_m)
        + vec3<f32>(0.72, 0.26, 0.08) * smoothstep(0.55, 0.95, live_ash) * (1.0 - day_m * 0.6) * 0.55;

    let tot_cov = clamp(cloud_cov * mix(0.62, 0.78, is_above) + ash_cov * 0.72, 0.0, 0.86);
    let ash_w = clamp(ash_cov / max(cloud_cov + ash_cov, 0.001), 0.0, 1.0);
    let final_col = mix(cloud_col, ash_col, ash_w);
    return vec4<f32>(final_col, tot_cov);
}

@fragment
fn fs_macro_atmosphere_resolve(in: FullScreenOut) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.pos.xy);
    var hdr = textureLoad(hdr_tex, pix, 0).rgb;
    let depth = textureLoad(depth_tex, pix, 0);

    let ro = uni.cam_pos.xyz;
    let r_surf = uni.planet_cfg.y; // ~1304.0m
    let r_atmo = uni.planet_cfg.z; // ~1520.0m
    let r_cloud = max(uni.sun_dir.w, r_surf + 120.0);
    let exposure = uni.planet_cfg.w;
    let t = uni.cam_pos.w;

    let cam_r = length(ro);
    let cam_alt = max(cam_r - r_surf, 0.0);
    let sun = normalize(uni.sun_dir.xyz);

    let rd = ray_dir_from_ndc(in.ndc);
    var t_scene = 1e6;
    let is_sky = select(0.0, 1.0, depth >= 0.99995);
    if (is_sky < 0.5) {
        let p_hit = uni.inv_view_proj * vec4<f32>(in.ndc, depth, 1.0);
        let rel_hit = p_hit.xyz / p_hit.w;
        t_scene = length(rel_hit);
    }

    // 1. Live PlanetaryClimateNet Coriolis Cloud & Volcanic Ash Shell (`r_cloud ≈ 1,442 m`)
    //    Composite over the planet surface when viewed from Stratosphere/Orbit, or overhead when on Ground/Flight!
    let hit_cloud = intersect_sphere(ro, rd, r_cloud);
    var cloud_hit_t = -1.0;
    if (hit_cloud.x < hit_cloud.y && hit_cloud.y > 0.0) {
        let cand = select(hit_cloud.y, hit_cloud.x, hit_cloud.x > 0.0);
        if (cand > 1.0 && cand < t_scene) {
            cloud_hit_t = cand;
        }
    }

    let in_atmo_factor = 1.0 - smoothstep(35.0, 240.0, cam_alt);
    let is_above_clouds = smoothstep(r_cloud - 40.0, r_cloud + 80.0, cam_r);

    // Composite orbit/stratosphere cloud deck directly onto the planet disc BEFORE atmospheric limb inscatter
    if (cloud_hit_t > 0.0 && is_above_clouds > 0.02 && is_sky < 0.5) {
        let cp = ro + rd * cloud_hit_t;
        let c_rad = normalize(cp);
        let cl = eval_coriolis_cloud_layer(c_rad, sun, t, 1.0);
        // Grazing limb fade so clouds hug the sphere cleanly
        let ndv_c = abs(dot(c_rad, rd));
        let edge_fade = smoothstep(0.06, 0.28, ndv_c);
        let alpha_c = cl.a * is_above_clouds * edge_fade;
        hdr = mix(hdr, cl.rgb, alpha_c);
    }

    // 2. Seamless Ground-to-Orbit Physical Rayleigh + Mie Atmospheric Scattering
    let hit_atmo = intersect_sphere(ro, rd, r_atmo);
    if (hit_atmo.x < hit_atmo.y && hit_atmo.y > 0.0) {
        let t0 = max(hit_atmo.x, 0.0);
        let t1 = min(hit_atmo.y, t_scene);
        if (t1 > t0) {
            let steps = 18;
            let dt = (t1 - t0) / f32(steps);
            let h_atmo = max(r_atmo - r_surf, 10.0);
            let h_scale_r = h_atmo * 0.165;
            let h_scale_m = h_atmo * 0.075;

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

            let sky_gain = mix(1.05, 1.38, in_atmo_factor);
            let aerial_dist = smoothstep(45.0, 520.0, t_scene);
            let surf_gain = mix(0.14, 0.24 * aerial_dist, in_atmo_factor);
            let scatter_gain = mix(surf_gain, sky_gain, is_sky);

            let inscatter = (sum_r * beta_r * phase_r * 1.25 + sum_m * beta_m * phase_m * 0.34) * scatter_gain;
            let trans_surf = exp(-(beta_r * od_r * mix(0.10, 0.14 * aerial_dist, in_atmo_factor) + beta_m * od_m * 0.12));
            let trans_sky = exp(-(beta_r * od_r * 1.15 + beta_m * od_m * 0.90));

            let sky_luma = dot(inscatter, vec3<f32>(0.2126, 0.7152, 0.0722));
            let day_cam = smoothstep(-0.05, 0.25, dot(normalize(ro), sun));
            let space_factor = smoothstep(130.0, 280.0, cam_alt);
            let star_keep = clamp(1.0 - sky_luma * 4.2, 0.0, 1.0) * mix(1.0 - day_cam, 1.0, space_factor);
            let bg_mult = mix(trans_surf, trans_sky * star_keep, is_sky);

            let p_mid = ro + rd * (0.5 * (t0 + t1));
            let mid_r = length(p_mid);
            let limb_f = exp(-max(mid_r - r_surf, 0.0) / (h_atmo * 0.20))
                * smoothstep(r_atmo, r_atmo - h_atmo * 0.25, mid_r);
            let term_mid = dot(normalize(p_mid), sun);
            let dusk_glow = exp(-pow((term_mid - 0.0) * 5.2, 2.0)) * limb_f
                * vec3<f32>(1.0, 0.36, 0.09) * mix(0.08, 0.56, is_sky) * (1.0 - 0.6 * in_atmo_factor);

            hdr = hdr * bg_mult + inscatter + dusk_glow;

            // Ground / Low Flight sky clouds: live PlanetaryClimateNet overhead cloud deck & drifting ash wisps!
            if (is_sky > 0.5 && cloud_hit_t > 0.0 && cloud_hit_t < 1150.0 && in_atmo_factor > 0.05) {
                let cp = ro + rd * cloud_hit_t;
                let c_rad = normalize(cp);
                let cl = eval_coriolis_cloud_layer(c_rad, sun, t, 0.0);
                let dist_fade = smoothstep(1150.0, 260.0, cloud_hit_t) * in_atmo_factor;
                hdr = mix(hdr, cl.rgb * 1.05, cl.a * dist_fade * 0.78);
            }
        }
    }

    let mapped = aces_filmic(hdr * exposure);
    let srgb = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
