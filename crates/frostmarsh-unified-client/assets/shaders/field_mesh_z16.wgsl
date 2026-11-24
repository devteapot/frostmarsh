// Wave 17B: Live True 3D Z=16..32 Smooth Surface-Nets Terrain Shader (AAA Stylized Overhaul)
// Renders directly into Approach4Renderer's shared 2xSS `ss_color_view` (Rgba8Unorm) + `ss_depth_view`!
// Features:
//   - Analytical hardware trilinear ∇solid_density smooth normals from `z16_density_tex`
//   - Fast unrolled 6-tap 3D volumetric sun-ray shadow march in `density_tex` (zero aliasing on overhangs,
//     arched bridges, cliffs & caves!) + 1-tap dynamic hero/shrine shadow map + 4-tap 3D voxel contact AO
//   - Rich triplanar Nordic rock/cliff stratification, snow-dusted upward ledges (N.y > 0.72),
//     carved Nordic ashlar stone-block relief on bridge spans, molten magma fissures (thermal > 0.55),
//     and cool turquoise/amethyst mineral crystal shimmer inside enclosed caverns (y < 6m)
//   - Live `Y`-key client-side visual style swap (`0 = NORDIC`, `1 = VOLCANIC`, `2 = ARCANE`, `3 = EMERALD`)

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
    vol_dims: vec4<f32>,      // xyz = (W, D, H), w = style_mode (0=NORDIC, 1=VOLCANIC, 2=ARCANE, 3=EMERALD)
    brick_dims: vec4<f32>,    // xyz = (occ_bx, occ_by, occ_bz), w = 8.0
    world_offset: vec4<f32>,  // xyz = (W*0.5, 0.0, H*0.5), w = stylized (1.0=PIXEL-2XSS, 0.0=CLEAN-3D)
    sky_top: vec4<f32>,       // rgb = linear sky top, w = sim_time
    sky_horizon: vec4<f32>,   // rgb = linear sky horizon, w = exposure
    hero_world_pos: vec4<f32>, // xyz = local player 3D world pos, w = cutaway enabled (0.0..1.0)
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<uniform> z16: Z16Uniforms;
@group(0) @binding(2) var density_tex: texture_3d<f32>;
@group(0) @binding(3) var albedo_tex: texture_3d<f32>;
@group(0) @binding(4) var brick_tex: texture_3d<f32>;
@group(0) @binding(5) var vol_samp: sampler;
@group(0) @binding(6) var shadow_tex: texture_depth_2d;
@group(0) @binding(7) var shadow_samp: sampler_comparison;

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
        return mix(bb, tint * 1.4, 0.35) * (t * t * 3.8);
    }
    if (style_id == 3) {
        let low = vec3<f32>(0.06, 0.64, 0.42);
        let mid = vec3<f32>(0.24, 0.98, 0.64);
        let hi  = vec3<f32>(0.84, 1.00, 0.90);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.35, 0.35) * (t * t * 3.6);
    }
    let low = vec3<f32>(0.98, 0.16, 0.02);
    let mid = vec3<f32>(1.00, 0.56, 0.08);
    let hi  = vec3<f32>(1.00, 0.96, 0.78);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    return mix(bb, tint * 1.35, 0.38) * (t * t * 3.9);
}

fn bayer4x4_threshold(pix: vec2<u32>) -> f32 {
    let m = array<f32, 16>(
         0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
        12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
         3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
        15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0,
    );
    let x = pix.x & 3u;
    let y = pix.y & 3u;
    return m[y * 4u + x];
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

// Fast 4-tap 3D voxel contact & crevice ambient occlusion in `density_tex`
fn compute_ao_3d(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let ref_v = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let t = normalize(cross(n, ref_v));

    var occ = 0.0;
    occ += sample_density(p + n * 0.72).r * 0.42;
    occ += sample_density(p + n * 1.95).r * 0.30;
    occ += sample_density(p + normalize(n + t * 0.80) * 1.85).r * 0.18;
    occ += sample_density(p + vec3<f32>(0.0, 3.6, 0.0)).r * 0.28;
    return clamp(1.0 - occ * 0.92, 0.14, 1.0);
}

// Fast 1-tap dynamic prop/hero shadow map comparison for grounded shadows on bridges & hills
fn sample_prop_shadow(rel_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    let biased = rel_pos + n * 0.16 + u.sun_dir.xyz * 0.12;
    let lclip = u.light_vp * vec4<f32>(biased, 1.0);
    let ndc = lclip.xyz / lclip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let in_b = uv.x > 0.01 && uv.x < 0.99 && uv.y > 0.01 && uv.y < 0.99 && ndc.z > 0.0 && ndc.z < 1.0;
    let s = textureSampleCompareLevel(shadow_tex, shadow_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), ndc.z - u.viewport.w * 1.2);
    return select(1.0, s, in_b);
}

// Unrolled 6-tap 3D volumetric sun-ray shadow march in `density_tex`:
// Crisp, alias-free overhang, arched-bridge, cliff & cave occlusion at ultra-low GPU cost!
fn compute_sun_shadow_3d(p: vec3<f32>, n: vec3<f32>, rel_pos: vec3<f32>) -> f32 {
    let l = u.sun_dir.xyz;
    let ndl = dot(n, l);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let dims = z16.vol_dims.xyz;
    let origin = p + n * 0.42;
    let tap_scale = clamp(dims.y / 16.0, 1.0, 2.5);
    let taps = array<f32, 6>(
        0.95,
        2.15,
        4.0 * mix(1.0, tap_scale, 0.35),
        6.8 * mix(1.0, tap_scale, 0.65),
        11.2 * tap_scale,
        16.8 * tap_scale,
    );

    var vis = 1.0;
    for (var i = 0; i < 6; i = i + 1) {
        let sp = origin + l * taps[i];
        if (sp.y >= dims.y || sp.x <= 0.0 || sp.x >= dims.x || sp.z <= 0.0 || sp.z >= dims.z) {
            break;
        }
        let d = sample_density(sp).r;
        // Soft analytical penumbra from trilinear solid isosurface (0.50)
        let pen = smoothstep(0.54, 0.22, d);
        vis = min(vis, pen);
        if (vis < 0.04) {
            vis = 0.0;
            break;
        }
    }
    let prop_s = sample_prop_shadow(rel_pos, n);
    let combined = min(vis, prop_s);
    let strength = clamp(u.sun_col.w * 1.12, 0.0, 0.88);
    return mix(1.0, combined, strength);
}

fn compute_emissive_bounce_3d(p: vec3<f32>, n: vec3<f32>, style_id: i32) -> vec3<f32> {
    var bounce = vec3<f32>(0.0);
    let dirs = array<vec3<f32>, 4>(
        vec3<f32>(0.984, 0.177, 0.0),
        vec3<f32>(-0.984, 0.177, 0.0),
        vec3<f32>(0.0, 0.177, 0.984),
        vec3<f32>(0.0, 0.177, -0.984),
    );
    let p_off = p + n * 0.65;
    for (var i = 0; i < 4; i = i + 1) {
        let dir = dirs[i];
        let s1 = sample_albedo(p_off + dir * 3.4);
        let w1 = max(dot(n, dir) * 0.55 + 0.45, 0.22);
        if (s1.a > 0.06) {
            bounce += blackbody_ramp(s1.a, srgb_to_linear(s1.rgb), style_id) * w1 * 0.62;
        }
    }
    return bounce;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.vol_pos;
    let wp = in.rel_pos + u.origin_time.xyz;
    let to_p = in.rel_pos - u.cam_pos.xyz;
    let hit_t = length(to_p);
    let rd = to_p / max(hit_t, 1e-4);

    // -------------------------------------------------------------------------
    // Wave 11C: Smart 3D Cave & Under-Bridge Ceiling Cutaway (BG3 / Diablo IV style)
    // -------------------------------------------------------------------------
    var cutaway_rim = 0.0;
    let cutaway_on = z16.hero_world_pos.w;
    if (cutaway_on > 0.01) {
        let hero_p = z16.hero_world_pos.xyz;
        if (wp.y > hero_p.y + 2.2) {
            let cam_w = u.cam_pos.xyz + u.origin_time.xyz;
            let focus_p = hero_p + vec3<f32>(0.0, 1.1, 0.0);
            let seg = focus_p - cam_w;
            let seg_len2 = max(dot(seg, seg), 1e-4);
            let t_seg = dot(wp - cam_w, seg) / seg_len2;
            if (t_seg > 0.02 && t_seg < 1.03) {
                let tc = clamp(t_seg, 0.0, 1.0);
                let closest = cam_w + seg * tc;
                let perp_d = length(wp - closest);
                let cone_r = mix(6.1, 5.0, tc);
                let h_fade = smoothstep(hero_p.y + 2.2, hero_p.y + 3.0, wp.y);
                let r_fade = 1.0 - smoothstep(cone_r * 0.56, cone_r, perp_d);
                let cut_alpha = r_fade * h_fade * cutaway_on;
                let thresh = bayer4x4_threshold(vec2<u32>(in.clip_pos.xy));
                if (cut_alpha > thresh + 0.02) {
                    discard;
                }
                let outer_rim = 1.0 - smoothstep(cone_r * 0.80, cone_r + 0.95, perp_d);
                let inner_rim = smoothstep(cone_r * 0.45, cone_r * 0.78, perp_d);
                cutaway_rim = outer_rim * inner_rim * h_fade * cutaway_on;
            }
        }
    }

    let dm = sample_density(p);
    let at = sample_albedo(p); // Pre-styled by `neural_style_z16.wgsl` 4 KB SRAM WGSL compute pass!

    var rough = clamp(dm.a, 0.08, 1.0);
    let n_base = sample_solid_normal(p);
    var alb_srgb = clamp(at.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var glow = at.a;
    let raw_thermal = at.a;
    let style_id = i32(round(z16.vol_dims.w));
    let stylized = z16.world_offset.w;
    let sim_t = z16.sky_top.w;
    let d_scale = max(z16.vol_dims.y / 16.0, 1.0);
    let h_norm = p.y / max(z16.vol_dims.y, 1.0);

    let raw_luma = dot(alb_srgb, vec3<f32>(0.299, 0.587, 0.114));
    let is_warm_wood = smoothstep(0.06, 0.18, alb_srgb.r - alb_srgb.b) * smoothstep(0.62, 0.36, raw_luma);
    let is_green_turf = smoothstep(0.04, 0.14, alb_srgb.g - max(alb_srgb.r, alb_srgb.b));

    if (style_id == 1) {
        rough = mix(rough, 0.38, smoothstep(0.26, 0.10, h_norm) * 0.55);
    } else if (style_id == 2) {
        rough = mix(rough, 0.28, 0.35);
    } else if (style_id == 3) {
        rough = mix(rough, 0.30, 0.32);
    }

    // -------------------------------------------------------------------------
    // Triplanar Painterly Surface Micro-Detail, Nordic Strata & Carved Masonry
    // -------------------------------------------------------------------------
    let macro_n = triplanar_fbm(p, n_base, 0.22);
    let detail_n = triplanar_noise(p, n_base, 1.35);
    let fine_n = vnoise2(p.xz * 3.80 + vec2<f32>(p.y * 1.9, -p.y * 1.4));

    let bump_x = vnoise2((p.xz + vec2<f32>(0.22, 0.0)) * 1.35) - detail_n;
    let bump_y = vnoise2((p.xy + vec2<f32>(0.0, 0.22)) * 1.35) - detail_n;
    let bump_z = vnoise2((p.xz + vec2<f32>(0.0, 0.22)) * 1.35) - detail_n;
    var n = normalize(n_base - vec3<f32>(bump_x, bump_y, bump_z) * (0.28 + 0.34 * rough));

    alb_srgb = clamp(
        alb_srgb * (0.85 + macro_n * 0.22 + (detail_n - 0.5) * 0.16 + (fine_n - 0.5) * 0.08),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let luma_s = dot(alb_srgb, vec3<f32>(0.299, 0.587, 0.114));
    let steep = smoothstep(0.76, 0.30, abs(n_base.y));
    let bridge_band = smoothstep(0.20, 0.28, h_norm) * smoothstep(0.62, 0.50, h_norm);
    // Detect neutral grey/slate stonework vs green turf/brown wood
    let sat_rg = abs(alb_srgb.r - alb_srgb.g) + abs(alb_srgb.g - alb_srgb.b);
    let is_stone_like = smoothstep(0.28, 0.10, sat_rg) * (1.0 - is_green_turf);

    // (1) Rich Triplanar Geological Rock/Cliff Stratification + Carved Nordic Ashlar Masonry
    if (steep > 0.04) {
        let strata_warp = vnoise2(p.xz * 0.55) * 2.8 + macro_n * 1.4;
        let strata_lo = sin(p.y * 3.8 + strata_warp);
        let strata_hi = sin(p.y * 9.6 + strata_warp * 1.7) * 0.45;
        let strata = (strata_lo * 0.65 + strata_hi * 0.35) * 0.16;

        // Carved Nordic stone-block courses on bridge arches & stone walls
        let course_y = abs(fract(p.y * 0.85) - 0.5);
        let row_shift = floor(p.y * 0.85) * 0.5;
        let course_x = abs(fract((p.x + p.z) * 0.45 + row_shift) - 0.5);
        let mortar = smoothstep(0.41, 0.49, max(course_y, course_x));
        let bevel = smoothstep(0.34, 0.43, max(course_y, course_x)) * (1.0 - mortar);

        let arch_w = bridge_band * (0.45 + 0.55 * is_stone_like);
        let warm_ledge = vec3<f32>(0.05, 0.04, 0.02) * max(strata_lo, 0.0);
        let cool_crev = vec3<f32>(-0.04, -0.02, 0.04) * max(-strata_lo, 0.0);
        let stone_mod = 1.0 + strata + bevel * 0.10 * arch_w - mortar * 0.26 * arch_w;

        alb_srgb = clamp(
            alb_srgb * mix(1.0, stone_mod, steep * 0.90) + (warm_ledge + cool_crev) * steep,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        n = normalize(n + vec3<f32>(strata * 0.38, -mortar * 0.32 * arch_w + bevel * 0.16 * arch_w, -strata * 0.38) * steep);
    }

    // (1B) Carved Nordic Stone-Block Flagstone Paving on Bridge Spans (top deck)
    let under_open = 1.0 - sample_density(p - vec3<f32>(0.0, 2.2, 0.0)).r;
    let is_bridge_deck = bridge_band * smoothstep(0.68, 0.88, n_base.y) * smoothstep(0.25, 0.72, under_open) * (1.0 - is_green_turf);
    if (is_bridge_deck > 0.05) {
        let bx = abs(fract(p.x * 0.55 + floor(p.z * 0.55) * 0.5) - 0.5);
        let bz = abs(fract(p.z * 0.55) - 0.5);
        let flag_mortar = smoothstep(0.43, 0.495, max(bx, bz));
        let block_var = (hash2(floor(p.xz * 0.55)) - 0.5) * 0.14;
        let deck_tint = vec3<f32>(0.54, 0.58, 0.66) * (0.94 + block_var);
        alb_srgb = mix(alb_srgb, deck_tint * (1.0 - flag_mortar * 0.28), is_bridge_deck * 0.58);
        n = normalize(n + vec3<f32>((bx - 0.25) * flag_mortar * 0.25, 0.0, (bz - 0.25) * flag_mortar * 0.25) * is_bridge_deck);
    }

    // (2) Warm wood grain & weathered plank seams on horizontal timber bridge decks
    let wood_deck = is_warm_wood * smoothstep(0.55, 0.82, n_base.y);
    if (wood_deck > 0.05) {
        let grain = sin(p.z * 11.0 + vnoise2(p.xz * 2.8) * 3.2) * 0.09;
        let plank_seam = smoothstep(0.90, 0.99, abs(sin(p.x * 2.4)));
        alb_srgb = clamp(alb_srgb * (1.0 + (grain - plank_seam * 0.22) * wood_deck), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // (3) Snow-Dusted Upward-Facing Alpine Ledges (`N.y > 0.72`, normalized `p.y > vol_dims.y * 0.45`) + Sparkle!
    // Keeps lowland tundra/moss & bridge decks richly textured while dusting alpine ridges & Floating Sky-Islands!
    let roof_step = clamp(d_scale, 1.0, 3.0);
    let overhead_rock = sample_density(p + vec3<f32>(0.0, 2.6 * roof_step, 0.0)).r
                      + sample_density(p + vec3<f32>(0.0, 4.8 * roof_step, 0.0)).r
                      + select(0.0, sample_density(p + vec3<f32>(0.0, 8.5, 0.0)).r, z16.vol_dims.y >= 48.0);
    let exposed_sky = 1.0 - smoothstep(0.30, 0.90, overhead_rock);
    if (style_id == 0) {
        let snow_ledge = smoothstep(0.72, 0.88, n_base.y)
                       * smoothstep(0.45, 0.68, h_norm + (macro_n - 0.5) * 0.18)
                       * exposed_sky
                       * (1.0 - is_bridge_deck * 0.92)
                       * (1.0 - wood_deck * 0.92)
                       * (1.0 - smoothstep(0.16, 0.40, glow));
        let high_snow = smoothstep(0.64, 0.84, luma_s) * smoothstep(0.45, 0.64, h_norm) * smoothstep(0.55, 0.82, n_base.y);
        let snow_w = clamp(max(snow_ledge * 0.62, high_snow * 0.72), 0.0, 1.0);
        if (snow_w > 0.04) {
            let speck = hash2(floor(p.xz * 7.2) + floor(p.y * 4.5));
            let sparkle = step(0.90, speck) * 0.22 * snow_w;
            let snow_col = vec3<f32>(0.76, 0.85, 0.95) + vec3<f32>(sparkle);
            alb_srgb = mix(alb_srgb, clamp(snow_col, vec3<f32>(0.0), vec3<f32>(1.0)), snow_w * 0.56);
            rough = mix(rough, 0.28, snow_w * 0.45);
        }
    } else {
        let snow_w = smoothstep(0.60, 0.80, luma_s) * smoothstep(0.45, 0.62, h_norm) * smoothstep(0.40, 0.80, n_base.y);
        if (snow_w > 0.05) {
            let speck = hash2(floor(p.xz * 6.5) + floor(p.y * 4.0));
            let sparkle = step(0.91, speck) * 0.22 * snow_w;
            alb_srgb = clamp(alb_srgb + vec3<f32>(sparkle), vec3<f32>(0.0), vec3<f32>(1.0));
            rough = mix(rough, 0.28, snow_w * 0.45);
        }
    }

    // (4) Wet shoreline sheen on low-roughness wet stone/banks
    let wet_w = smoothstep(0.50, 0.22, rough);
    if (wet_w > 0.05) {
        let pebble = (fine_n - 0.5) * 0.14;
        alb_srgb = clamp(alb_srgb * (0.94 + pebble * wet_w), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // (5) Warm Glowing Magma Fissures when thermal > 0.55 (or Volcanic hotspots!)
    var magma_emit = vec3<f32>(0.0);
    if (raw_thermal > 0.45 || glow > 0.05 || (style_id == 1 && h_norm < 0.42)) {
        let crack_n = triplanar_fbm(p + vec3<f32>(0.0, -sim_t * 0.16, 0.0), n_base, 1.55);
        let crack_sharp = 1.0 - smoothstep(0.015, 0.085, abs(crack_n - 0.50));
        let pulse = 0.82 + 0.18 * sin(sim_t * 3.4 + p.x * 0.5 + p.z * 0.5);
        let hot_fissure = smoothstep(0.56, 0.78, raw_thermal);
        let crack_boost = select(0.0, crack_sharp * 0.24 * smoothstep(0.32, 0.08, h_norm), style_id == 1);
        if (hot_fissure > 0.02) {
            // Charred obsidian edge framing glowing molten veins
            let char_rim = smoothstep(0.14, 0.06, abs(crack_n - 0.50)) * (1.0 - crack_sharp);
            alb_srgb = mix(alb_srgb, vec3<f32>(0.09, 0.06, 0.08), char_rim * hot_fissure * 0.65);
            magma_emit = blackbody_ramp(clamp(0.55 + crack_sharp * 0.45, 0.0, 1.0), vec3<f32>(1.0, 0.48, 0.08), style_id)
                       * crack_sharp * hot_fissure * pulse * 1.45;
        }
        glow = clamp(glow + crack_boost + glow * crack_sharp * 0.42, 0.0, 1.0);
    }

    // (6) Arcane Thermal holographic contour scanlines along elevation strata in Style 2!
    if (style_id == 2) {
        let contour = smoothstep(0.91, 0.99, abs(sin(p.y * 3.14159)));
        let grid_line = smoothstep(0.96, 0.995, max(abs(sin(p.x * 0.7854)), abs(sin(p.z * 0.7854))));
        alb_srgb = clamp(alb_srgb + vec3<f32>(0.05, 0.30, 0.40) * (contour * 0.60 + grid_line * 0.22), vec3<f32>(0.0), vec3<f32>(1.0));
        glow = max(glow, contour * 0.22 + grid_line * 0.08);
    }

    // (7) Emerald Cavern bioluminescent jade mineral veins in Style 3!
    if (style_id == 3) {
        let vein_n = triplanar_fbm(p * 1.15 + vec3<f32>(sim_t * 0.04, 0.0, -sim_t * 0.03), n_base, 1.10);
        let vein = 1.0 - smoothstep(0.02, 0.09, abs(vein_n - 0.52));
        let pulse = 0.75 + 0.25 * sin(sim_t * 1.8 + p.x * 0.25 + p.z * 0.25);
        alb_srgb = clamp(alb_srgb + vec3<f32>(0.04, 0.32, 0.20) * vein * pulse, vec3<f32>(0.0), vec3<f32>(1.0));
        glow = max(glow, vein * 0.34 * pulse);
    }

    let albedo = srgb_to_linear(alb_srgb);
    let ao = compute_ao_3d(p, n_base);
    let shadow = compute_sun_shadow_3d(p, n_base, in.rel_pos);
    let l = u.sun_dir.xyz;
    let v = -rd;
    let ndl = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);

    var light_scalar = ndl * shadow;
    if (stylized > 0.5) {
        let q_l = round(light_scalar * 9.0) / 9.0;
        light_scalar = mix(light_scalar, q_l, 0.52);
    }

    let shininess = mix(96.0, 12.0, rough);
    let spec_strength = mix(0.74, 0.05, rough);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let spec = pow(ndh, shininess) * spec_strength * (0.4 + 0.6 * fresnel) * ndl * shadow;

    let sky_w = n.y * 0.5 + 0.5;
    var ambient_bot = vec3<f32>(0.045, 0.068, 0.135);
    if (style_id == 1) {
        ambient_bot = vec3<f32>(0.14, 0.05, 0.03);
    } else if (style_id == 2) {
        ambient_bot = vec3<f32>(0.04, 0.08, 0.20);
    } else if (style_id == 3) {
        ambient_bot = vec3<f32>(0.03, 0.13, 0.10);
    }
    let ambient = mix(ambient_bot, z16.sky_top.rgb * 0.82 + z16.sky_horizon.rgb * 0.34, sky_w) * ao;
    let direct = u.sun_col.rgb * light_scalar * 1.02;
    let bounce_emit = compute_emissive_bounce_3d(p, n_base, style_id);

    // Subterranean vaulted cavern lighting + Cool Turquoise / Amethyst Crystal Shimmer (`p.y < vol_dims.y * 0.28`)!
    let in_cave = smoothstep(0.32, 1.05, overhead_rock)
                * smoothstep(z16.vol_dims.y * 0.28, z16.vol_dims.y * 0.10, p.y);
    var cave_lit = vec3<f32>(0.0);
    if (in_cave > 0.02) {
        let cave_albedo = max(albedo, vec3<f32>(0.085, 0.095, 0.125));
        let torch_dir = normalize(u.cam_pos.xyz - in.rel_pos + vec3<f32>(0.0, 0.8, 0.0));
        let torch_att = 1.0 / (1.0 + hit_t * hit_t * 0.0022);
        let torch_nd = max(dot(n, torch_dir), 0.15);
        let cave_bounce_col = select(vec3<f32>(0.92, 0.42, 0.16), vec3<f32>(0.22, 0.88, 0.56), style_id == 3);
        let magma_glow_amb = cave_bounce_col * (0.26 + 0.48 * max(-n.y, 0.0) + 0.34 * torch_nd * torch_att);
        let cavern_rim = vec3<f32>(0.22, 0.38, 0.66) * (0.24 + 0.65 * pow(1.0 - ndv, 2.0));

        // Faceted glowing turquoise & amethyst subterranean mineral veins!
        let cryst_n = triplanar_fbm(p * 0.85, n_base, 1.25);
        let cryst_vein = 1.0 - smoothstep(0.015, 0.075, abs(cryst_n - 0.52));
        let hue_sel = smoothstep(0.35, 0.65, triplanar_noise(p, n_base, 0.28));
        let turq = vec3<f32>(0.10, 0.92, 0.84);
        let amet = vec3<f32>(0.68, 0.24, 1.00);
        let cryst_col = mix(turq, amet, hue_sel);
        let cryst_pulse = 0.72 + 0.28 * sin(sim_t * 2.2 + p.x * 0.35 + p.z * 0.42);
        let cryst_glint = pow(ndh, 28.0) * 0.8 + fresnel * 0.65;
        let crystal_emit = cryst_col * cryst_vein * (0.55 + cryst_glint) * cryst_pulse;

        cave_lit = (cave_albedo * (magma_glow_amb + cavern_rim + bounce_emit * 2.2) + crystal_emit * 1.15) * in_cave;
    }

    // Upper Sky-Island cloud-lit mist & underside sky-bounce illumination (`p.y > vol_dims.y * 0.62`, Z>=48)
    var sky_island_lit = vec3<f32>(0.0);
    if (z16.vol_dims.y >= 48.0 && h_norm > 0.58) {
        let sky_w_alt = smoothstep(0.58, 0.82, h_norm);
        let underside = max(-n.y, 0.0) * 0.55 + pow(1.0 - ndv, 2.2) * 0.38;
        let mist_col = mix(z16.sky_horizon.rgb, z16.sky_top.rgb * 1.35 + u.sun_col.rgb * 0.32, 0.45);
        sky_island_lit = albedo * mist_col * (0.20 + underside * 0.58) * sky_w_alt;
    }

    var hdr = albedo * (direct + ambient * 0.78 + bounce_emit) + cave_lit + sky_island_lit + magma_emit + u.sun_col.rgb * spec;

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
        var cool_ink = srgb_to_linear(vec3<f32>(0.149, 0.118, 0.282));
        var warm_hi = srgb_to_linear(vec3<f32>(1.000, 0.957, 0.839));
        if (style_id == 1) {
            cool_ink = srgb_to_linear(vec3<f32>(0.18, 0.05, 0.08));
            warm_hi = srgb_to_linear(vec3<f32>(1.00, 0.78, 0.44));
        } else if (style_id == 2) {
            cool_ink = srgb_to_linear(vec3<f32>(0.05, 0.08, 0.28));
            warm_hi = srgb_to_linear(vec3<f32>(0.76, 0.98, 1.00));
        } else if (style_id == 3) {
            cool_ink = srgb_to_linear(vec3<f32>(0.04, 0.16, 0.15));
            warm_hi = srgb_to_linear(vec3<f32>(0.80, 1.00, 0.86));
        }
        let shadow_t = clamp(1.0 - light_scalar * 1.38, 0.0, 1.0) * (1.0 - clamp(glow * 2.0, 0.0, 1.0));
        hdr = mix(hdr, cool_ink * (0.24 + 0.56 * ao), shadow_t * 0.20);
        let hi_t = clamp((light_scalar - 0.68) * 1.8, 0.0, 1.0);
        hdr = mix(hdr, hdr * 0.84 + warm_hi * 0.24, hi_t * 0.26);
    }

    if (glow > 0.04) {
        hdr += blackbody_ramp(glow, albedo, style_id);
    }

    // Soft glowing rock cross-section rim along Smart 3D Cave/Bridge Ceiling Cutaway edge!
    if (cutaway_rim > 0.01) {
        var rim_col = vec3<f32>(0.68, 0.86, 1.05);
        if (style_id == 1) {
            rim_col = vec3<f32>(1.15, 0.56, 0.20);
        } else if (style_id == 2) {
            rim_col = vec3<f32>(0.42, 0.92, 1.25);
        } else if (style_id == 3) {
            rim_col = vec3<f32>(0.26, 1.08, 0.66);
        }
        hdr += rim_col * cutaway_rim * 0.65;
    }

    // Distance atmospheric perspective fog matching the active style sky (scaled for tall Z=64 vistas!)
    let fog_start = 68.0 * clamp(z16.vol_dims.y / 24.0, 1.0, 2.2);
    let fog_span = 190.0 * clamp(z16.vol_dims.y / 24.0, 1.0, 2.0);
    let fog_t = clamp((hit_t - fog_start) / fog_span, 0.0, 0.64);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t * (1.0 - clamp(glow, 0.0, 0.85)));

    return vec4<f32>(aces_tonemap(hdr * z16.sky_horizon.w), 1.0);
}
