// Iteration 10A: Live True 3D Z=16 Smooth Surface-Nets Mesh Shader
// Renders directly into Approach4Renderer's shared 2xSS `ss_color_view` (Rgba8Unorm) + `ss_depth_view`!
// Features:
//   - Analytical hardware trilinear ∇solid_density smooth normals from `z16_density_tex`
//   - 3D-texture soft cavern & crevice AO + 3D-texture brick-accelerated raymarched sun shadows
//   - Triplanar painterly masonry bevels, rock strata, weathered wood planks, alpine snow sparkle,
//     wet shoreline sheen & molten ember crack veins
//   - Live `Y`-key client-side visual style swap (`0 = NORDIC`, `1 = VOLCANIC EMBER`, `2 = ARCANE THERMAL`)
//     with zero server changes!
//   - 9-step Cool-Indigo (#261e48) / Warm-Cream (#fff4d6) stylized cel ramp + 8 dynamic point lights

struct SceneUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,
    cam_fwd: vec4<f32>,
    origin_time: vec4<f32>,
    viewport: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_col: vec4<f32>,
    sky_col: vec4<f32>,
    bounce_col: vec4<f32>,
    fog_col: vec4<f32>,
    point_pos: array<vec4<f32>, 8>,
    point_col: array<vec4<f32>, 8>,
};

struct Z16Uniforms {
    inv_vp: mat4x4<f32>,
    vol_dims: vec4<f32>,      // xyz = (W, D, H), w = style_mode (0=NORDIC, 1=VOLCANIC, 2=ARCANE)
    brick_dims: vec4<f32>,    // xyz = (occ_bx, occ_by, occ_bz), w = 8.0
    world_offset: vec4<f32>,  // xyz = (W*0.5, 0.0, H*0.5), w = stylized (1.0=PIXEL-2XSS, 0.0=CLEAN-3D)
    sky_top: vec4<f32>,       // rgb = linear sky top, w = sim_time
    sky_horizon: vec4<f32>,   // rgb = linear sky horizon, w = exposure
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<uniform> z16: Z16Uniforms;
@group(0) @binding(2) var density_tex: texture_3d<f32>;
@group(0) @binding(3) var albedo_tex: texture_3d<f32>;
@group(0) @binding(4) var brick_tex: texture_3d<f32>;
@group(0) @binding(5) var vol_samp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>, // volume space [0..W, 0..D, 0..H]
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) vol_pos: vec3<f32>,
    @location(1) rel_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_p = in.pos - z16.world_offset.xyz;
    let rel_p = world_p - u.origin_time.xyz;
    var out: VsOut;
    out.clip_pos = u.vp * vec4<f32>(rel_p, 1.0);
    out.vol_pos = in.pos;
    out.rel_pos = rel_p;
    return out;
}

fn world_to_uvw(p: vec3<f32>) -> vec3<f32> {
    return p / z16.vol_dims.xyz;
}

fn sample_density(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(density_tex, vol_samp, world_to_uvw(p), 0.0);
}

fn sample_albedo(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(albedo_tex, vol_samp, world_to_uvw(p), 0.0);
}

fn sample_solid_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.28;
    let gx = sample_density(p + vec3<f32>(e, 0.0, 0.0)).r - sample_density(p - vec3<f32>(e, 0.0, 0.0)).r;
    let gy = sample_density(p + vec3<f32>(0.0, e, 0.0)).r - sample_density(p - vec3<f32>(0.0, e, 0.0)).r;
    let gz = sample_density(p + vec3<f32>(0.0, 0.0, e)).r - sample_density(p - vec3<f32>(0.0, 0.0, e)).r;
    let g = vec3<f32>(gx, gy, gz);
    let len = length(g);
    if (len < 1e-4) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return -g / len;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let mapped = clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u2 = f * f * (vec2<f32>(3.0) - 2.0 * f);
    return mix(
        mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u2.x),
        mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u2.x),
        u2.y,
    );
}

fn fbm2(p: vec2<f32>) -> f32 {
    return vnoise2(p) * 0.58
         + vnoise2(p * 2.13 + vec2<f32>(13.7, -9.2)) * 0.28
         + vnoise2(p * 4.37 - vec2<f32>(5.1, 19.3)) * 0.14;
}

fn triplanar_noise(p: vec3<f32>, n: vec3<f32>, freq: f32) -> f32 {
    let w = pow(abs(n), vec3<f32>(3.0));
    let ws = w / max(w.x + w.y + w.z, 1e-4);
    return vnoise2(p.yz * freq) * ws.x
         + vnoise2(p.xz * freq) * ws.y
         + vnoise2(p.xy * freq) * ws.z;
}

fn triplanar_fbm(p: vec3<f32>, n: vec3<f32>, freq: f32) -> f32 {
    let w = pow(abs(n), vec3<f32>(3.0));
    let ws = w / max(w.x + w.y + w.z, 1e-4);
    return fbm2(p.yz * freq) * ws.x
         + fbm2(p.xz * freq) * ws.y
         + fbm2(p.xy * freq) * ws.z;
}

fn blackbody_ramp(temp: f32, tint: vec3<f32>, style_id: i32) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
    if (style_id == 2) {
        let low = vec3<f32>(0.15, 0.78, 1.00);
        let mid = vec3<f32>(1.00, 0.20, 0.74);
        let hi  = vec3<f32>(1.00, 0.96, 0.80);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.4, 0.35) * (t * t * 3.6);
    }
    let low = vec3<f32>(0.95, 0.14, 0.02);
    let mid = vec3<f32>(1.00, 0.52, 0.08);
    let hi  = vec3<f32>(1.00, 0.94, 0.72);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    return mix(bb, tint * 1.35, 0.42) * (t * t * 3.6);
}

fn eval_sky(rd: vec3<f32>) -> vec3<f32> {
    let h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    let up = pow(clamp(rd.y, 0.0, 1.0), 0.55);
    var sky = mix(z16.sky_horizon.rgb, z16.sky_top.rgb, up);
    if (rd.y < 0.0) {
        sky = mix(z16.sky_horizon.rgb, z16.sky_horizon.rgb * 0.55, clamp(-rd.y * 4.0, 0.0, 1.0));
    }
    let sun_dot = max(dot(rd, u.sun_dir.xyz), 0.0);
    let sun_halo = pow(sun_dot, 14.0) * 0.28 + pow(sun_dot, 140.0) * 1.4;
    return sky + u.sun_col.rgb * sun_halo * smoothstep(-0.05, 0.18, h);
}

fn compute_ao_3d(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let ref_v = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let t = normalize(cross(n, ref_v));
    let b = cross(n, t);

    var occ = 0.0;
    occ += sample_density(p + n * 0.75).r * 0.30;
    occ += sample_density(p + n * 1.85).r * 0.24;
    occ += sample_density(p + n * 3.60).r * 0.16;
    occ += sample_density(p + normalize(n + t * 0.75) * 2.2).r * 0.12;
    occ += sample_density(p + normalize(n - t * 0.75) * 2.2).r * 0.12;
    occ += sample_density(p + normalize(n + b * 0.75) * 2.6).r * 0.12;
    occ += sample_density(p + normalize(n - b * 0.75) * 2.6).r * 0.12;
    occ += sample_density(p + vec3<f32>(0.0, 4.2, 0.0)).r * 0.22;
    return clamp(1.0 - occ * 0.88, 0.10, 1.0);
}

fn compute_sun_shadow_3d(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let l = u.sun_dir.xyz;
    let ndl = dot(n, l);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let dims = z16.vol_dims.xyz;
    let bdims = vec3<i32>(z16.brick_dims.xyz);
    let bsize = z16.brick_dims.w;

    var t = 0.65;
    var vis = 1.0;
    for (var i = 0; i < 24; i = i + 1) {
        let sp = p + n * 0.35 + l * t;
        if (sp.y >= dims.y || sp.x <= 0.0 || sp.x >= dims.x || sp.z <= 0.0 || sp.z >= dims.z) {
            break;
        }
        let bc = clamp(vec3<i32>(floor(sp / bsize)), vec3<i32>(0), bdims - vec3<i32>(1));
        let occ = textureLoad(brick_tex, bc, 0);
        if (occ.r < 0.04) {
            t += 2.2;
            continue;
        }
        let d = sample_density(sp).r;
        if (d >= 0.5) {
            return 0.0;
        }
        let margin = max(0.5 - d, 0.0);
        vis = min(vis, clamp(margin * 4.5 + 0.12, 0.0, 1.0));
        t += 0.55 + t * 0.08;
    }
    return mix(1.0, vis, clamp(u.sun_col.w, 0.0, 1.0));
}

fn compute_emissive_bounce_3d(p: vec3<f32>, n: vec3<f32>, style_id: i32) -> vec3<f32> {
    var bounce = vec3<f32>(0.0);
    let dirs = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.18, 0.0),
        vec3<f32>(-1.0, 0.18, 0.0),
        vec3<f32>(0.0, 0.18, 1.0),
        vec3<f32>(0.0, 0.18, -1.0),
    );
    let p_off = p + n * 0.65;
    for (var i = 0; i < 4; i = i + 1) {
        let dir = normalize(dirs[i]);
        let s1 = sample_albedo(p_off + dir * 2.8);
        let s2 = sample_albedo(p_off + dir * 7.2);
        let w1 = max(dot(n, dir) * 0.55 + 0.45, 0.22);
        if (s1.a > 0.06) {
            bounce += blackbody_ramp(s1.a, srgb_to_linear(s1.rgb), style_id) * w1 * 0.45;
        }
        if (s2.a > 0.06) {
            bounce += blackbody_ramp(s2.a, srgb_to_linear(s2.rgb), style_id) * w1 * 0.30;
        }
    }
    return bounce;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.vol_pos;
    let to_p = in.rel_pos - u.cam_pos.xyz;
    let hit_t = length(to_p);
    let rd = to_p / max(hit_t, 1e-4);

    let dm = sample_density(p);
    let at = sample_albedo(p);

    var rough = clamp(dm.a, 0.08, 1.0);
    let n_base = sample_solid_normal(p);
    var alb_srgb = clamp(at.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var glow = at.a;
    let style_id = i32(round(z16.vol_dims.w));
    let stylized = z16.world_offset.w;
    let sim_t = z16.sky_top.w;

    // -------------------------------------------------------------------------
    // Live Client Style Swap Transform (`Y`-key: 0=NORDIC, 1=VOLCANIC, 2=ARCANE)
    // Purely client-side visual reinterpretation of the live physical 3D volume!
    // -------------------------------------------------------------------------
    let raw_luma = dot(alb_srgb, vec3<f32>(0.299, 0.587, 0.114));
    let is_greenish = smoothstep(0.02, 0.12, alb_srgb.g - max(alb_srgb.r, alb_srgb.b));
    let is_warm_wood = smoothstep(0.06, 0.18, alb_srgb.r - alb_srgb.b) * smoothstep(0.62, 0.40, raw_luma);
    let is_icy = smoothstep(0.60, 0.80, raw_luma) * smoothstep(0.02, 0.12, alb_srgb.b - alb_srgb.r);

    if (style_id == 0) {
        // Style 0: NORDIC — crisp emerald boreal moss, warm timber, cool granite, & high-peak alpine snow!
        let ridge_snow = smoothstep(9.1, 10.8, p.y) * smoothstep(0.42, 0.78, n_base.y) * (1.0 - is_warm_wood);
        alb_srgb = mix(alb_srgb, vec3<f32>(0.86, 0.91, 0.97), ridge_snow * 0.78);
        // Subtle vibrant boost on Nordic moss & weathered stone
        let moss_boost = vec3<f32>(alb_srgb.r * 0.92, min(alb_srgb.g * 1.14, 1.0), alb_srgb.b * 0.94);
        alb_srgb = mix(alb_srgb, moss_boost, is_greenish * 0.65);
    } else if (style_id == 1) {
        // Style 1: VOLCANIC EMBER — obsidian basalt, scorched ash-ochre, forged iron/charred timber, molten veins!
        let obsidian = vec3<f32>(0.15, 0.12, 0.14) * (0.75 + raw_luma * 0.75);
        let scorched = vec3<f32>(0.42, 0.22, 0.14) * (0.85 + raw_luma * 0.45);
        let forged = vec3<f32>(0.34, 0.20, 0.14);
        let ash_peak = vec3<f32>(0.46, 0.40, 0.38);
        var volc = mix(obsidian, scorched, is_greenish * 0.85);
        volc = mix(volc, forged, is_warm_wood * 0.85);
        volc = mix(volc, ash_peak, smoothstep(9.2, 11.4, p.y) * 0.65 + is_icy * 0.5);
        alb_srgb = volc;
        rough = mix(rough, 0.38, smoothstep(3.8, 1.6, p.y) * 0.55);
    } else {
        // Style 2: ARCANE THERMAL — twilight-indigo crystalline slate, teal-violet runes & glowing thermal strata!
        let slate = vec3<f32>(0.11, 0.15, 0.30) * (0.78 + raw_luma * 0.85);
        let biolum_moss = vec3<f32>(0.12, 0.44, 0.58);
        let rune_gold = vec3<f32>(0.44, 0.32, 0.68);
        let crystal_peak = vec3<f32>(0.54, 0.84, 0.98);
        var arc = mix(slate, biolum_moss, is_greenish * 0.80);
        arc = mix(arc, rune_gold, is_warm_wood * 0.80);
        arc = mix(arc, crystal_peak, smoothstep(9.2, 11.4, p.y) * 0.65 + is_icy * 0.6);
        alb_srgb = arc;
        rough = mix(rough, 0.28, 0.35);
    }

    // -------------------------------------------------------------------------
    // Triplanar Painterly Surface Micro-Detail (zero flat plastic!)
    // -------------------------------------------------------------------------
    let macro_n = triplanar_fbm(p, n_base, 0.24);
    let detail_n = triplanar_noise(p, n_base, 1.35);
    let fine_n = triplanar_noise(p, n_base, 3.80);

    let d_x = triplanar_noise(p + vec3<f32>(0.22, 0.0, 0.0), n_base, 1.35) - detail_n;
    let d_y = triplanar_noise(p + vec3<f32>(0.0, 0.22, 0.0), n_base, 1.35) - detail_n;
    let d_z = triplanar_noise(p + vec3<f32>(0.0, 0.0, 0.22), n_base, 1.35) - detail_n;
    var n = normalize(n_base - vec3<f32>(d_x, d_y, d_z) * (0.28 + 0.35 * rough));

    alb_srgb = clamp(
        alb_srgb * (0.86 + macro_n * 0.20 + (detail_n - 0.5) * 0.14 + (fine_n - 0.5) * 0.06),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let luma_s = dot(alb_srgb, vec3<f32>(0.299, 0.587, 0.114));
    let steep = smoothstep(0.76, 0.34, abs(n_base.y));

    // (1) Stone masonry bevels & geological rock strata on steep faces, arches & cave walls
    if (steep > 0.04) {
        let strata = sin(p.y * 4.4 + vnoise2(p.xz * 0.65) * 2.4) * 0.11;
        let course_y = abs(fract(p.y * 0.85) - 0.5);
        let row_shift = floor(p.y * 0.85) * 0.5;
        let course_x = abs(fract((p.x + p.z) * 0.42 + row_shift) - 0.5);
        let mortar = smoothstep(0.42, 0.49, max(course_y, course_x));
        let arch_band = smoothstep(3.0, 4.2, p.y) * smoothstep(9.5, 8.0, p.y);
        let stone_mod = 1.0 + strata - mortar * 0.18 * arch_band;
        alb_srgb = clamp(alb_srgb * mix(1.0, stone_mod, steep * 0.85), vec3<f32>(0.0), vec3<f32>(1.0));
        n = normalize(n + vec3<f32>(strata * 0.26, -mortar * 0.22 * arch_band, -strata * 0.26) * steep);
    }

    // (2) Warm wood grain & weathered plank seams on horizontal timber bridge decks
    let wood_deck = is_warm_wood * smoothstep(0.55, 0.82, n_base.y);
    if (wood_deck > 0.05) {
        let grain = sin(p.z * 11.0 + vnoise2(p.xz * 2.8) * 3.2) * 0.08;
        let plank_seam = smoothstep(0.90, 0.99, abs(sin(p.x * 2.4)));
        alb_srgb = clamp(alb_srgb * (1.0 + (grain - plank_seam * 0.20) * wood_deck), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // (3) Alpine snow crystalline sparkle on bright upper ledges
    let snow_w = smoothstep(0.60, 0.80, luma_s) * smoothstep(7.8, 9.6, p.y) * smoothstep(0.40, 0.80, n_base.y);
    if (snow_w > 0.05) {
        let speck = hash2(floor(p.xz * 6.5) + floor(p.y * 4.0));
        let sparkle = step(0.91, speck) * 0.22 * snow_w;
        alb_srgb = clamp(alb_srgb + vec3<f32>(sparkle), vec3<f32>(0.0), vec3<f32>(1.0));
        rough = mix(rough, 0.28, snow_w * 0.45);
    }

    // (4) Wet shoreline sheen on low-roughness wet stone/banks
    let wet_w = smoothstep(0.50, 0.22, rough);
    if (wet_w > 0.05) {
        let pebble = (fine_n - 0.5) * 0.14;
        alb_srgb = clamp(alb_srgb * (0.94 + pebble * wet_w), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // (5) Molten ember cracks & glowing lava veins on hot / Volcanic hotspots!
    if (glow > 0.05 || (style_id == 1 && p.y < 5.6)) {
        let crack_n = triplanar_fbm(p + vec3<f32>(0.0, -sim_t * 0.15, 0.0), n_base, 1.45);
        let vein = smoothstep(0.43, 0.51, abs(crack_n - 0.5));
        let crack_boost = select(0.0, (1.0 - vein) * 0.36 * smoothstep(5.6, 1.8, p.y), style_id == 1);
        glow = clamp(glow + crack_boost + glow * (1.0 - vein) * 0.35, 0.0, 1.0);
    }

    // (6) Arcane Thermal holographic contour scanlines along elevation strata in Style 2!
    if (style_id == 2) {
        let contour = smoothstep(0.91, 0.99, abs(sin(p.y * 3.14159)));
        let grid_line = smoothstep(0.96, 0.995, max(abs(sin(p.x * 0.7854)), abs(sin(p.z * 0.7854))));
        alb_srgb = clamp(alb_srgb + vec3<f32>(0.05, 0.30, 0.40) * (contour * 0.60 + grid_line * 0.22), vec3<f32>(0.0), vec3<f32>(1.0));
        glow = max(glow, contour * 0.22 + grid_line * 0.08);
    }

    let albedo = srgb_to_linear(alb_srgb);
    let ao = compute_ao_3d(p, n_base);
    let shadow = compute_sun_shadow_3d(p, n_base);
    let l = u.sun_dir.xyz;
    let v = -rd;
    let ndl = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);

    var light_scalar = ndl * shadow;
    if (stylized > 0.5) {
        let q_l = round(light_scalar * 9.0) / 9.0;
        light_scalar = mix(light_scalar, q_l, 0.55);
    }

    let shininess = mix(96.0, 10.0, rough);
    let spec_strength = mix(0.68, 0.04, rough);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let spec = pow(ndh, shininess) * spec_strength * (0.4 + 0.6 * fresnel) * ndl * shadow;

    let sky_w = n.y * 0.5 + 0.5;
    let ambient_bot = select(
        select(vec3<f32>(0.04, 0.06, 0.12), vec3<f32>(0.14, 0.05, 0.03), style_id == 1),
        vec3<f32>(0.04, 0.08, 0.20),
        style_id == 2
    );
    let ambient = mix(ambient_bot, z16.sky_top.rgb * 0.85 + z16.sky_horizon.rgb * 0.35, sky_w) * ao;
    let direct = u.sun_col.rgb * light_scalar;
    let bounce_emit = compute_emissive_bounce_3d(p, n_base, style_id);

    // Subterranean vaulted cavern lighting (warm magma bounce + cool basalt rim so caves never crush to black)
    let overhead_rock = sample_density(p + vec3<f32>(0.0, 2.8, 0.0)).r
                      + sample_density(p + vec3<f32>(0.0, 4.6, 0.0)).r;
    let in_cave = smoothstep(0.40, 1.15, overhead_rock) * smoothstep(6.2, 3.8, p.y);
    let cave_albedo = max(albedo, vec3<f32>(0.085, 0.095, 0.125));
    let torch_dir = normalize(u.cam_pos.xyz - in.rel_pos + vec3<f32>(0.0, 0.8, 0.0));
    let torch_att = 1.0 / (1.0 + hit_t * hit_t * 0.0022);
    let torch_nd = max(dot(n, torch_dir), 0.15);
    let magma_glow_amb = vec3<f32>(0.92, 0.42, 0.16) * (0.28 + 0.52 * max(-n.y, 0.0) + 0.35 * torch_nd * torch_att);
    let cavern_rim = vec3<f32>(0.22, 0.36, 0.62) * (0.22 + 0.65 * pow(1.0 - ndv, 2.0));
    let cave_lit = cave_albedo * (magma_glow_amb + cavern_rim + bounce_emit * 2.2) * in_cave;

    var hdr = albedo * (direct + ambient * 0.82 + bounce_emit) + cave_lit + u.sun_col.rgb * spec;

    // 8 dynamic point lights (live wildfires, spells, shrines, golden relic, time domes)
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lp = u.point_pos[i];
        if (lp.w > 0.0) {
            let d3 = lp.xyz - in.rel_pos;
            let r = length(d3);
            if (r < lp.w) {
                let att = max(0.0, 1.0 - r / lp.w);
                let l_ndl = max(dot(n, d3 / max(r, 0.001)), 0.0);
                hdr += (albedo * 0.90 + vec3<f32>(0.04)) * srgb_to_linear(u.point_col[i].rgb) * att * att * (0.40 + 0.60 * l_ndl) * 1.4;
            }
        }
    }

    // Signature Cool-Indigo (#261e48) shadow pull + Warm-Cream (#fff4d6) highlight push!
    if (stylized > 0.5) {
        let cool_ink = select(
            select(srgb_to_linear(vec3<f32>(0.149, 0.118, 0.282)), srgb_to_linear(vec3<f32>(0.18, 0.05, 0.08)), style_id == 1),
            srgb_to_linear(vec3<f32>(0.05, 0.08, 0.28)),
            style_id == 2
        );
        let warm_hi = select(
            select(srgb_to_linear(vec3<f32>(1.000, 0.957, 0.839)), srgb_to_linear(vec3<f32>(1.00, 0.78, 0.44)), style_id == 1),
            srgb_to_linear(vec3<f32>(0.76, 0.98, 1.00)),
            style_id == 2
        );
        let shadow_t = clamp(1.0 - light_scalar * 1.45, 0.0, 1.0) * (1.0 - clamp(glow * 2.0, 0.0, 1.0));
        hdr = mix(hdr, cool_ink * (0.25 + 0.55 * ao), shadow_t * 0.16);
        let hi_t = clamp((light_scalar - 0.72) * 1.8, 0.0, 1.0);
        hdr = mix(hdr, hdr * 0.82 + warm_hi * 0.22, hi_t * 0.24);
    }

    if (glow > 0.04) {
        hdr += blackbody_ramp(glow, albedo, style_id);
    }

    // Distance atmospheric perspective fog matching the active style sky
    let fog_t = clamp((hit_t - 65.0) / 185.0, 0.0, 0.70);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t * (1.0 - clamp(glow, 0.0, 0.85)));

    return vec4<f32>(aces_tonemap(hdr * z16.sky_horizon.w), 1.0);
}
