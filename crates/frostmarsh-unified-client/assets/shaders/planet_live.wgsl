// Wave 31B: Live Curved 6-Face Spherical Planetary Metal Shader (`planet_live.wgsl`)
//
// Pipelines:
//   1. vs_fullscreen + fs_starfield          -> Deep cosmic velvet starfield, nebula band & live Sun disc
//   2. vs_planet     + fs_planet_solid       -> Curved 6-face terrain, polar ice caps, molten volcanic rift,
//                                               Orbital Sky-Ring Islands & Buoyant Crystal Cores, 60 live 3D
//                                               heroes, 6 shrines, & solid combat projectiles
//   3. vs_planet     + fs_planet_ocean       -> Fresnel sapphire curved ocean with waves, foam & sea-ice
//   4. vs_planet     + fs_planet_translucent -> Depth-tested volumetric steam/smoke plumes, shrine light
//                                               pillars, hero aura rings & elemental actuator beams
//   5. vs_fullscreen + fs_atmosphere_resolve -> Depth-aware physical Rayleigh + Mie atmospheric limb halo
//                                               (sunlit sapphire + amber-crimson twilight terminator) + ACES

struct PlanetUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,    // xyz = camera world pos, w = sim_time
    sun_dir: vec4<f32>,    // xyz = normalized sun dir, w = r_ocean
    planet_cfg: vec4<f32>, // x = r_core, y = r_surf_atmo, z = r_atmo_outer, w = exposure
    viewport: vec4<f32>,   // x = width, y = height, z = 1/width, w = 1/height
};

@group(0) @binding(0) var<uniform> uni: PlanetUniforms;

// -----------------------------------------------------------------------------
// Seamless 3D Spherical Procedural Noise (Zero UV Seams Across All 6 Faces)
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

    // Deep cosmic indigo-black velvet + sweeping galactic dust nebula band
    var col = vec3<f32>(0.003, 0.005, 0.013);
    let band = exp(-pow(dot(rd, normalize(vec3<f32>(0.28, 0.85, -0.44))) * 4.2, 2.0));
    let neb = fbm3(rd * 4.5 + vec3<f32>(2.1, 0.7, -1.3));
    col += band * vec3<f32>(0.019, 0.028, 0.056) * smoothstep(0.32, 0.82, neb);
    col += band * vec3<f32>(0.026, 0.013, 0.040) * smoothstep(0.55, 0.90, fbm3(rd * 8.0 + 4.0));

    // Fine pinpoint starfield with subtle twinkle
    let g1 = rd * 240.0;
    let c1 = floor(g1);
    let f1 = fract(g1) - 0.5;
    let h1 = hash31(c1);
    if (h1 > 0.9915) {
        let jitter = vec3<f32>(hash31(c1 + 7.1), hash31(c1 + 19.3), hash31(c1 + 37.9)) - 0.5;
        let d = length(f1 - jitter * 0.55);
        let tw = 0.86 + 0.14 * sin(t * (1.8 + fract(h1 * 37.0) * 3.0) + h1 * 90.0);
        let b = smoothstep(0.32, 0.04, d) * (0.55 + 1.45 * fract(h1 * 137.0)) * tw;
        let tint = mix(vec3<f32>(0.72, 0.84, 1.0), vec3<f32>(1.0, 0.90, 0.72), fract(h1 * 53.0));
        col += tint * b * 1.45;
    }

    // Bright hero navigation stars
    let g2 = rd * 135.0;
    let c2 = floor(g2);
    let f2 = fract(g2) - 0.5;
    let h2 = hash31(c2 + 113.0);
    if (h2 > 0.9962) {
        let jitter2 = vec3<f32>(hash31(c2 + 11.7), hash31(c2 + 29.1), hash31(c2 + 43.3)) - 0.5;
        let d2 = length(f2 - jitter2 * 0.45);
        let tw2 = 0.88 + 0.12 * sin(t * (1.2 + fract(h2 * 19.0) * 2.5) + h2 * 71.0);
        let b2 = smoothstep(0.36, 0.03, d2) * (1.2 + 2.2 * fract(h2 * 91.0)) * tw2;
        let tint2 = mix(vec3<f32>(0.68, 0.86, 1.0), vec3<f32>(1.0, 0.82, 0.62), fract(h2 * 29.0));
        col += tint2 * b2 * 1.85;
    }

    // Crisp distant solar disc + warm solar corona
    let sun_cos = max(dot(rd, sun), 0.0);
    let sun_disc = smoothstep(0.9992, 0.99985, sun_cos);
    let sun_halo = pow(sun_cos, 48.0) * 0.48 + pow(sun_cos, 6.0) * 0.035;
    col += vec3<f32>(1.0, 0.94, 0.82) * sun_disc * 9.5 + vec3<f32>(1.0, 0.78, 0.52) * sun_halo;

    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Shared Planet Vertex I/O (`PlanetVertex`: pos, normal, color, flags)
//   color: rgb = base albedo (or glow tint), a = roughness (solid) or alpha (translucent)
//   flags: x = water_depth, y = emissive_strength, z = mat_kind, w = ice_frac
//     mat_kind:
//       0.0 = Curved Planetary Terrain / Continents / Ice / Lava Rift / Sky-Island Crust
//       1.0 = Curved Specular/Fresnel Ocean
//       2.0 = Pulsing Buoyant Crystal Core (Orbital Sky-Ring)
//       3.0 = Live 3D Hero / Creature Avatar / Planetary Shrine Obelisk
//       4.0 = Live Elemental Beam / Projectile / Luminous Beacon
//       5.0 = Volumetric Steam / Smoke Plume
// -----------------------------------------------------------------------------
struct VertIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) flags: vec4<f32>,
};

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) flags: vec4<f32>,
};

@vertex
fn vs_planet(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos = uni.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.normal = in.normal;
    out.color = in.color;
    out.flags = in.flags;
    return out;
}

// -----------------------------------------------------------------------------
// Pass 1B: Curved Terrain + Orbital Sky-Ring + Crystal Cores + Live 3D Heroes/Shrines
// -----------------------------------------------------------------------------
@fragment
fn fs_planet_solid(in: VertOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(uni.cam_pos.xyz - p);
    let t = uni.cam_pos.w;

    let mat_kind = in.flags.z;
    let em_strength = max(in.flags.y, 0.0);
    let rough = clamp(in.color.a, 0.08, 0.96);

    // Day/Night Terminator illumination driven by live sweeping sun_dir
    let term_raw = dot(radial, sun);
    let day_mask = smoothstep(-0.14, 0.22, term_raw);
    let twilight = exp(-pow((term_raw - 0.02) * 5.2, 2.0));
    let sun_col = mix(
        vec3<f32>(1.0, 0.46, 0.16),
        vec3<f32>(1.06, 0.98, 0.88),
        smoothstep(-0.04, 0.30, term_raw),
    );

    // Luminous combat projectiles / beam cores (mat_kind >= 3.5)
    if (mat_kind > 3.5) {
        let pulse = 0.90 + 0.16 * sin(t * 7.5 + length(p) * 0.4);
        let ndv = max(dot(normalize(in.normal), view_dir), 0.0);
        let core = mix(0.75, 1.45, pow(ndv, 1.4));
        let lit = in.color.rgb * (0.4 + em_strength * pulse * core);
        return vec4<f32>(lit, 1.0);
    }

    // Live 3D Heroes, Creatures & Planetary Shrine Obelisks (mat_kind ~ 3.0)
    if (mat_kind > 2.5) {
        let n = normalize(in.normal);
        let ndl = max(dot(n, sun), 0.0);
        let wrap_ndl = max((dot(n, sun) + 0.28) / 1.28, 0.0);
        let sky_up = 0.5 + 0.5 * dot(n, radial);
        let amb = mix(vec3<f32>(0.08, 0.12, 0.22), vec3<f32>(0.24, 0.34, 0.52), sky_up)
            * mix(0.65, 1.0, day_mask);
        let diff = sun_col * (ndl * 1.45 * day_mask + wrap_ndl * 0.42 * twilight + 0.22 * (1.0 - day_mask));

        let half_v = normalize(sun + view_dir);
        let shininess = mix(72.0, 12.0, rough);
        let spec = pow(max(dot(n, half_v), 0.0), shininess) * mix(0.55, 0.12, rough) * (0.35 + 0.65 * day_mask);

        // Crisp team-colored Fresnel rim so 60 heroes read cleanly even on the night hemisphere!
        let fres = pow(1.0 - max(dot(n, view_dir), 0.0), 2.4);
        let rim = in.color.rgb * fres * mix(0.72, 0.32, day_mask);
        let pulse = 0.92 + 0.12 * sin(t * 4.5 + p.x * 0.3 + p.z * 0.3);
        let emissive = in.color.rgb * em_strength * pulse * mix(1.45, 1.05, day_mask);

        let lit = in.color.rgb * (diff + amb) + sun_col * spec + rim + emissive;
        return vec4<f32>(lit, 1.0);
    }

    // Pulsing Buoyant Crystal Core voxels on the Orbital Sky-Ring (mat_kind ~ 2.0)
    if (mat_kind > 1.5) {
        let n = normalize(in.normal);
        let pulse = 0.84 + 0.22 * sin(t * 3.6 + dot(p, vec3<f32>(0.18, 0.14, 0.22)));
        let ndv = max(dot(n, view_dir), 0.0);
        let fres = pow(1.0 - ndv, 2.0);
        let crystal_core = mix(vec3<f32>(0.08, 0.72, 1.0), vec3<f32>(0.68, 0.22, 1.0), fres * 0.72);
        let emissive = crystal_core * max(em_strength, 0.85) * pulse * mix(1.25, 0.88, day_mask);
        let lit = in.color.rgb * (0.20 + 0.45 * day_mask) + emissive;
        return vec4<f32>(lit, 1.0);
    }

    // Standard Curved Planetary Terrain + Volcanic Rift + Polar Glacier Caps (mat_kind ~ 0.0)
    let micro = fbm3(p * 0.68);
    let micro_fine = noise3(p * 2.8);
    let detail_mod = 0.80 + 0.32 * micro + 0.10 * (micro_fine - 0.5);

    var n = normalize(in.normal);
    let eps = 0.20;
    let gx = fbm3((p + vec3<f32>(eps, 0.0, 0.0)) * 0.95) - fbm3((p - vec3<f32>(eps, 0.0, 0.0)) * 0.95);
    let gy = fbm3((p + vec3<f32>(0.0, eps, 0.0)) * 0.95) - fbm3((p - vec3<f32>(0.0, eps, 0.0)) * 0.95);
    let gz = fbm3((p + vec3<f32>(0.0, 0.0, eps)) * 0.95) - fbm3((p - vec3<f32>(0.0, 0.0, eps)) * 0.95);
    let grad = vec3<f32>(gx, gy, gz);
    n = normalize(n - (grad - radial * dot(grad, radial)) * 0.85);

    let slope = clamp(1.0 - max(dot(n, radial), 0.0), 0.0, 1.0);
    let rock_col = vec3<f32>(0.30, 0.31, 0.35) * (0.85 + 0.30 * micro_fine);
    let is_lava = step(0.25, em_strength);
    let base_albedo = mix(
        in.color.rgb,
        rock_col,
        smoothstep(0.20, 0.48, slope) * (1.0 - is_lava) * 0.68,
    );

    let ndl = max(dot(n, sun), 0.0);
    let wrap_ndl = max((dot(n, sun) + 0.14) / 1.14, 0.0);
    let diffuse = sun_col * (ndl * 1.55 * day_mask + wrap_ndl * 0.34 * twilight);

    let sky_up = 0.5 + 0.5 * dot(n, radial);
    let day_amb = mix(vec3<f32>(0.03, 0.05, 0.09), vec3<f32>(0.12, 0.21, 0.38), sky_up)
        * (0.20 + 0.80 * day_mask);
    // Readable cool lunar ambient on night side so continents & mountains keep silhouette
    let night_amb = vec3<f32>(0.015, 0.024, 0.048) * (1.0 - day_mask);

    let half_v = normalize(sun + view_dir);
    let shininess = mix(96.0, 6.0, rough);
    let spec = pow(max(dot(n, half_v), 0.0), shininess) * mix(0.42, 0.04, rough) * day_mask;

    let fres = pow(1.0 - max(dot(n, view_dir), 0.0), 3.4);
    let rim_col = mix(vec3<f32>(0.10, 0.32, 0.72), vec3<f32>(0.95, 0.42, 0.15), twilight)
        * fres
        * (0.18 * day_mask + 0.26 * twilight);

    // Incandescent magma / lava rift glow blazing vividly across the night hemisphere
    let lava_pulse = 0.82 + 0.38 * smoothstep(0.22, 0.86, micro + 0.08 * sin(t * 2.2));
    let lava_tint = mix(
        vec3<f32>(1.0, 0.24, 0.03),
        vec3<f32>(1.0, 0.76, 0.24),
        clamp(em_strength * 0.45, 0.0, 1.0),
    );
    let emissive = lava_tint * em_strength * lava_pulse * mix(1.75, 1.05, day_mask);

    let albedo = base_albedo * detail_mod;
    let lit = albedo * (diffuse + day_amb + night_amb) + sun_col * spec + rim_col + emissive;
    return vec4<f32>(lit, 1.0);
}

// -----------------------------------------------------------------------------
// Pass 1C: Curved Specular/Fresnel Sapphire Ocean & Live Sea-Ice Pass
// -----------------------------------------------------------------------------
@fragment
fn fs_planet_ocean(in: VertOut) -> @location(0) vec4<f32> {
    let water_depth = in.flags.x;
    if (water_depth < 0.06) {
        discard;
    }
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let view_dir = normalize(uni.cam_pos.xyz - p);

    let ice_frac = clamp(in.flags.w, 0.0, 1.0);
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
    let sun_col = mix(
        vec3<f32>(1.0, 0.46, 0.16),
        vec3<f32>(1.05, 0.98, 0.90),
        smoothstep(-0.04, 0.30, term_raw),
    );

    // Depth-based ocean color: turquoise shallows -> royal sapphire -> abyssal navy
    let shallow_col = vec3<f32>(0.03, 0.46, 0.64);
    let mid_col = vec3<f32>(0.014, 0.15, 0.44);
    let deep_col = vec3<f32>(0.006, 0.044, 0.20);
    let d_norm = clamp(water_depth / 2.2, 0.0, 1.0);
    var water_col = mix(shallow_col, mid_col, smoothstep(0.0, 0.38, d_norm));
    water_col = mix(water_col, deep_col, smoothstep(0.32, 1.0, d_norm));

    let foam = smoothstep(0.45, 0.06, water_depth) * smoothstep(0.30, 0.72, w1 * 0.6 + w2 * 0.4);
    water_col = mix(water_col, vec3<f32>(0.84, 0.94, 0.99), foam * 0.65);

    let ice_col = vec3<f32>(0.78, 0.91, 0.99) * (0.90 + 0.14 * w1);
    let base_col = mix(water_col, ice_col, ice_frac);

    let ndv = max(dot(n, view_dir), 0.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 4.4);
    let sky_refl = mix(vec3<f32>(0.016, 0.034, 0.078), vec3<f32>(0.18, 0.48, 0.95), day_mask)
        + vec3<f32>(1.0, 0.45, 0.16) * twilight * 0.70;

    let half_v = normalize(sun + view_dir);
    let spec_sharp = pow(max(dot(n, half_v), 0.0), mix(160.0, 28.0, ice_frac)) * mix(2.25, 0.4, ice_frac);
    let spec_broad = pow(max(dot(n, half_v), 0.0), 22.0) * mix(0.28, 0.08, ice_frac);
    let spec = sun_col * (spec_sharp + spec_broad) * (day_mask + twilight * 0.48);

    let ndl = max(dot(n, sun), 0.0);
    let diff = base_col * (
        sun_col * ndl * 1.25 * day_mask
            + vec3<f32>(0.07, 0.14, 0.28) * (0.20 + 0.80 * day_mask)
            + vec3<f32>(0.012, 0.022, 0.046) * (1.0 - day_mask)
    );

    let col = mix(diff, sky_refl, fresnel * mix(0.66, 0.22, ice_frac)) + spec;
    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// Pass 1D: Translucent Steam/Smoke Plumes, Shrine Light Pillars, Aura Rings & Beams
// -----------------------------------------------------------------------------
@fragment
fn fs_planet_translucent(in: VertOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let radial = normalize(p);
    let sun = normalize(uni.sun_dir.xyz);
    let t = uni.cam_pos.w;
    let mat_kind = in.flags.z;
    let em = max(in.flags.y, 0.0);
    let alpha = clamp(in.color.a, 0.0, 1.0);

    // Volumetric steam/smoke plumes (mat_kind ~ 5.0)
    if (mat_kind > 4.5) {
        let puff = fbm3(p * 0.55 - radial * (t * 0.5));
        let term_raw = dot(radial, sun);
        let day_mask = smoothstep(-0.18, 0.24, term_raw);
        let twilight = exp(-pow((term_raw - 0.02) * 4.5, 2.0));
        let sun_tint = mix(vec3<f32>(0.24, 0.30, 0.45), vec3<f32>(1.05, 0.98, 0.92), day_mask)
            + vec3<f32>(1.0, 0.48, 0.18) * twilight * 0.55;
        let col = in.color.rgb * sun_tint * (0.75 + 0.45 * puff) + in.color.rgb * em;
        let a = alpha * smoothstep(0.18, 0.80, puff);
        if (a < 0.02) {
            discard;
        }
        return vec4<f32>(col, a);
    }

    // Glowing Shrine Light Pillars, Hero Aura Rings & Elemental Actuator Beams
    let pulse = 0.88 + 0.18 * sin(t * 5.2 + length(p) * 0.35);
    let col = in.color.rgb * (0.65 + em * pulse);
    return vec4<f32>(col, alpha);
}

// -----------------------------------------------------------------------------
// Pass 2: Depth-Aware Physical Rayleigh + Mie Atmosphere & ACES Filmic Tonemap
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
    let r_surf = uni.planet_cfg.y;
    let r_atmo = uni.planet_cfg.z;
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
            let h_scale_r = max((r_atmo - r_surf) * 0.22, 0.5);
            let h_scale_m = max((r_atmo - r_surf) * 0.10, 0.25);

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
            let scatter_gain = mix(0.68, 1.78, is_sky);
            let inscatter = (sum_r * beta_r * phase_r * 1.42 + sum_m * beta_m * phase_m * 0.46) * scatter_gain;
            let trans = exp(-(beta_r * od_r * 0.30 + beta_m * od_m * 0.25));

            // Warm golden-amber twilight glow along the Day/Night Terminator limb
            let p_mid = ro + rd * (0.5 * (t0 + t1));
            let mid_r = length(p_mid);
            let limb_f = exp(-max(mid_r - r_surf, 0.0) / max((r_atmo - r_surf) * 0.24, 0.5));
            let term_mid = dot(normalize(p_mid), sun);
            let dusk_glow = exp(-pow((term_mid - 0.0) * 5.0, 2.0))
                * limb_f
                * vec3<f32>(1.0, 0.36, 0.09)
                * mix(0.14, 0.56, is_sky);

            hdr = hdr * mix(trans, vec3<f32>(1.0), is_sky * 0.85) + inscatter + dusk_glow;
        }
    }

    let mapped = aces_filmic(hdr * exposure);
    let srgb = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
