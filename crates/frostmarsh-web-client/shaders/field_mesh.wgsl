// Pass 1: Wave 18A AAA Stylized Smooth Surface-Nets 3D Isosurface Mesh Shader (WebGPU, Z=16..32)
// Features:
//   - Fast unrolled 6-tap 3D volumetric sun-ray shadow march + 4-tap 3D voxel contact AO in `density_tex`
//   - Client-side WGSL 4-Style Neural Compute Transform (`Y`-key):
//       0 = NORDIC, 1 = VOLCANIC EMBER, 2 = ARCANE THERMAL, 3 = EMERALD CAVERN
//   - Triplanar Painterly Surface Micro-Detail (zero flat plastic!):
//       * Multi-octave warped geological strata & carved Nordic ashlar stone-block masonry on arches/decks
//       * Weathered timber plank seams & wood grain on horizontal decks
//       * Snow-dusted upward ledges (`N.y > 0.72`) with crystalline alpine sparkle
//       * Wet shoreline pebble darkening & glossy Fresnel sheen
//       * Incandescent molten fissure glow (`thermal > 0.55`) with charred obsidian rims
//       * Cool cyan/turquoise & amethyst mineral crystal veins inside enclosed caves
//   - Smart 3D Cave Ceiling Cutaway (`C`-key Capsule Walk Mode) with luminous rock cross-section rim!

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,         // xyz = cam_pos, w = sim_time
    sun_dir: vec4<f32>,         // xyz = sun_dir, w = exposure
    sun_col: vec4<f32>,         // rgb = sun_col, w = style_mode (0..3)
    sky_top: vec4<f32>,         // rgb = sky_top, w = stylized (1.0 = Pixel-2xSS, 0.0 = Clean-3D)
    sky_horizon: vec4<f32>,     // rgb = sky_horizon, w = neural_style_strength
    vol_dims: vec4<f32>,        // xyz = (W, D, H), w = 1.0 / D
    brick_dims: vec4<f32>,      // xyz = (bx, by, bz), w = brick_size (8.0)
    viewport: vec4<f32>,        // xy = (w, h), zw = (1/w, 1/h)
    player_cutaway: vec4<f32>,  // xyz = capsule_pos, w = cutaway_enabled (1.0 in Capsule Walk)
    style_weights_a: vec4<f32>, // rgb = rock/base neural tint, w = glow_gain
    style_weights_b: vec4<f32>, // rgb = moss/foliage neural tint, w = rough_bias
    style_weights_c: vec4<f32>, // rgb = peak/snow/crystal neural tint, w = vein_gain
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var density_tex: texture_3d<f32>;
@group(0) @binding(2) var albedo_tex: texture_3d<f32>;
@group(0) @binding(3) var brick_tex: texture_3d<f32>;
@group(0) @binding(4) var vol_samp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    return out;
}

fn world_to_uvw(p: vec3<f32>) -> vec3<f32> {
    return p / u.vol_dims.xyz;
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

fn bayer4_cutaway(cell: vec2<i32>) -> f32 {
    var m = array<f32, 16>(
        0.0,  8.0,  2.0, 10.0,
        12.0, 4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
        15.0, 7.0, 13.0,  5.0
    );
    return (m[(cell.x & 3) + ((cell.y & 3) << 2u)] + 0.5) / 16.0;
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
        // Arcane Thermal: electric cyan -> magenta -> incandescent white-gold
        let low = vec3<f32>(0.14, 0.78, 1.00);
        let mid = vec3<f32>(1.00, 0.18, 0.74);
        let hi  = vec3<f32>(1.00, 0.96, 0.80);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.40, 0.36) * (t * t * 3.8);
    }
    if (style_id == 3) {
        // Emerald Cavern: deep viridian -> radiant jade -> mint-gold bioluminescence
        let low = vec3<f32>(0.04, 0.68, 0.36);
        let mid = vec3<f32>(0.22, 1.00, 0.60);
        let hi  = vec3<f32>(0.86, 1.00, 0.88);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.35, 0.32) * (t * t * 3.6);
    }
    let low = vec3<f32>(0.95, 0.14, 0.02);
    let mid = vec3<f32>(1.00, 0.52, 0.08);
    let hi  = vec3<f32>(1.00, 0.94, 0.72);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    return mix(bb, tint * 1.35, 0.44) * (t * t * 3.8);
}

fn eval_sky(rd: vec3<f32>) -> vec3<f32> {
    let h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    let up = pow(clamp(rd.y, 0.0, 1.0), 0.55);
    var sky = mix(u.sky_horizon.rgb, u.sky_top.rgb, up);
    if (rd.y < 0.0) {
        sky = mix(u.sky_horizon.rgb, u.sky_horizon.rgb * 0.55, clamp(-rd.y * 4.0, 0.0, 1.0));
    }
    let sun_dot = max(dot(rd, u.sun_dir.xyz), 0.0);
    let sun_halo = pow(sun_dot, 14.0) * 0.32 + pow(sun_dot, 180.0) * 1.8;
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

// Unrolled 6-tap 3D volumetric sun-ray shadow march in `density_tex`:
// Crisp, alias-free overhang, arched-bridge, cliff & cave occlusion at ultra-low GPU cost!
fn compute_sun_shadow_3d(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let l = u.sun_dir.xyz;
    let ndl = dot(n, l);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let dims = u.vol_dims.xyz;
    let origin = p + n * 0.42;
    let taps = array<f32, 6>(0.95, 2.15, 4.0, 6.8, 11.2, 16.8);

    var vis = 1.0;
    for (var i = 0; i < 6; i = i + 1) {
        let sp = origin + l * taps[i];
        if (sp.y >= dims.y || sp.x <= 0.0 || sp.x >= dims.x || sp.z <= 0.0 || sp.z >= dims.z) {
            break;
        }
        let d = sample_density(sp).r;
        let pen = smoothstep(0.54, 0.22, d);
        vis = min(vis, pen);
        if (vis < 0.04) {
            return 0.0;
        }
    }
    return vis;
}

fn compute_emissive_bounce_3d(p: vec3<f32>, n: vec3<f32>, style_id: i32) -> vec3<f32> {
    var bounce = vec3<f32>(0.0);
    let dirs = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 0.18, 0.0),
        vec3<f32>(-1.0, 0.18, 0.0),
        vec3<f32>(0.0, 0.18, 1.0),
        vec3<f32>(0.0, 0.18, -1.0),
        vec3<f32>(-0.7, -0.3, 0.65),
        vec3<f32>(-0.7, -0.3, -0.65)
    );
    let p_off = p + n * 0.65;
    for (var i = 0; i < 6; i = i + 1) {
        let dir = normalize(dirs[i]);
        let s1 = sample_albedo(p_off + dir * 2.8);
        let s2 = sample_albedo(p_off + dir * 6.8);
        let w1 = max(dot(n, dir) * 0.55 + 0.45, 0.22);
        if (s1.a > 0.06) {
            bounce += blackbody_ramp(s1.a, srgb_to_linear(s1.rgb), style_id) * w1 * 0.36;
        }
        if (s2.a > 0.06) {
            bounce += blackbody_ramp(s2.a, srgb_to_linear(s2.rgb), style_id) * w1 * 0.22;
        }
    }
    let s_low = sample_albedo(vec3<f32>(p.x, max(p.y - 3.2, 1.0), p.z));
    if (s_low.a > 0.08) {
        bounce += blackbody_ramp(s_low.a, srgb_to_linear(s_low.rgb), style_id) * 0.28;
    }
    return bounce;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;

    // -------------------------------------------------------------------------
    // Smart 3D Cave Ceiling Cutaway (`C`-key Capsule Walk Mode)
    // Smoothly Bayer-dithers away roof/ceiling fragments strictly above
    // `player_y + 2.2` inside the camera-to-player line-of-sight cone!
    // -------------------------------------------------------------------------
    var cutaway_rim = 0.0;
    if (u.player_cutaway.w > 0.5) {
        let player_p = u.player_cutaway.xyz;
        let ceil_y = player_p.y + 2.2;
        if (p.y > ceil_y) {
            let cam_p = u.cam_pos.xyz;
            let focus_p = player_p + vec3<f32>(0.0, 1.05, 0.0);
            let seg = focus_p - cam_p;
            let seg_len = length(seg);
            let seg_dir = seg / max(seg_len, 1e-4);
            let cam_to_frag = p - cam_p;
            let proj_t = dot(cam_to_frag, seg_dir);
            if (proj_t > 0.20 && proj_t < seg_len + 2.8) {
                let closest = cam_p + seg_dir * clamp(proj_t, 0.0, seg_len);
                let perp_dist = length(p - closest);
                let cone_t = clamp(proj_t / max(seg_len, 1.0), 0.0, 1.0);
                let cone_r = mix(1.35, 5.80, cone_t);
                let radial_fade = 1.0 - smoothstep(cone_r * 0.42, cone_r, perp_dist);
                let height_fade = smoothstep(ceil_y, ceil_y + 1.35, p.y);
                let cut_alpha = radial_fade * height_fade;
                if (cut_alpha > 0.02) {
                    let bayer_coord = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
                    if (cut_alpha > bayer4_cutaway(bayer_coord)) {
                        discard;
                    }
                }
                let outer_rim = 1.0 - smoothstep(cone_r * 0.78, cone_r + 0.90, perp_dist);
                let inner_rim = smoothstep(cone_r * 0.44, cone_r * 0.76, perp_dist);
                cutaway_rim = outer_rim * inner_rim * height_fade;
            }
        }
    }

    let to_p = p - u.cam_pos.xyz;
    let hit_t = length(to_p);
    let rd = to_p / max(hit_t, 1e-4);

    let dm = sample_density(p);
    let at = sample_albedo(p);

    var rough = clamp(dm.a, 0.08, 1.0);
    let n_base = sample_solid_normal(p);
    var alb_srgb = clamp(at.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var glow = at.a;
    let raw_thermal = at.a;
    let style_id = i32(round(u.sun_col.w));
    let stylized = u.sky_top.w;
    let sim_t = u.cam_pos.w;
    let d_scale = max(u.vol_dims.y / 16.0, 1.0);

    // -------------------------------------------------------------------------
    // Live Client-Side WGSL 4-Style Neural Compute Transform (`Y`-key)
    // Driven by neural style weights on `(albedo_thermal, density_mat, altitude, normal)`
    //   0 = NORDIC, 1 = VOLCANIC EMBER, 2 = ARCANE THERMAL, 3 = EMERALD CAVERN
    // -------------------------------------------------------------------------
    let raw_luma = dot(alb_srgb, vec3<f32>(0.299, 0.587, 0.114));
    let alt_norm = clamp(p.y * u.vol_dims.w, 0.0, 1.0);
    let is_greenish = smoothstep(0.02, 0.12, alb_srgb.g - max(alb_srgb.r, alb_srgb.b));
    let is_warm_wood = smoothstep(0.06, 0.18, alb_srgb.r - alb_srgb.b) * smoothstep(0.64, 0.38, raw_luma);
    let is_icy = smoothstep(0.60, 0.80, raw_luma) * smoothstep(0.02, 0.12, alb_srgb.b - alb_srgb.r);
    let high_peak = max(smoothstep(9.2, 11.4, p.y), smoothstep(0.62, 0.84, alt_norm));

    if (style_id == 0) {
        // Style 0: NORDIC — crisp emerald boreal moss, warm timber, cool granite, sparkling alpine snow
        let ridge_snow = high_peak * smoothstep(0.42, 0.78, n_base.y) * (1.0 - is_warm_wood);
        alb_srgb = mix(alb_srgb, u.style_weights_c.rgb, ridge_snow * 0.76);
        let moss_boost = vec3<f32>(alb_srgb.r * 0.92, min(alb_srgb.g * 1.14, 1.0), alb_srgb.b * 0.94);
        alb_srgb = mix(alb_srgb, moss_boost, is_greenish * 0.65);
    } else if (style_id == 1) {
        // Style 1: VOLCANIC EMBER — obsidian basalt, scorched ash-ochre, forged iron, molten ember veins
        let obsidian = u.style_weights_a.rgb * (0.72 + raw_luma * 0.82);
        let scorched = u.style_weights_b.rgb * (0.84 + raw_luma * 0.48);
        let forged = vec3<f32>(0.34, 0.20, 0.14);
        var volc = mix(obsidian, scorched, is_greenish * 0.85);
        volc = mix(volc, forged, is_warm_wood * 0.85);
        volc = mix(volc, u.style_weights_c.rgb, high_peak * 0.62 + is_icy * 0.48);
        alb_srgb = volc;
        rough = clamp(mix(rough, 0.36, smoothstep(4.0, 1.6, p.y) * 0.55) + u.style_weights_b.w, 0.08, 1.0);
    } else if (style_id == 2) {
        // Style 2: ARCANE THERMAL — twilight-indigo crystalline slate, teal-violet runes & thermal strata
        let slate = u.style_weights_a.rgb * (0.76 + raw_luma * 0.88);
        let biolum_moss = u.style_weights_b.rgb;
        let rune_gold = vec3<f32>(0.46, 0.32, 0.70);
        var arc = mix(slate, biolum_moss, is_greenish * 0.82);
        arc = mix(arc, rune_gold, is_warm_wood * 0.80);
        arc = mix(arc, u.style_weights_c.rgb, high_peak * 0.62 + is_icy * 0.58);
        alb_srgb = arc;
        rough = clamp(mix(rough, 0.28, 0.35) + u.style_weights_b.w, 0.08, 1.0);
    } else {
        // Style 3: EMERALD CAVERN — deep malachite limestone, bioluminescent jade moss & crystal flora
        let malachite = u.style_weights_a.rgb * (0.75 + raw_luma * 0.85);
        let jade_moss = u.style_weights_b.rgb * (0.85 + raw_luma * 0.40);
        let petrified_wood = vec3<f32>(0.24, 0.36, 0.28);
        var em = mix(malachite, jade_moss, is_greenish * 0.88);
        em = mix(em, petrified_wood, is_warm_wood * 0.80);
        em = mix(em, u.style_weights_c.rgb, high_peak * 0.58 + is_icy * 0.52);
        alb_srgb = em;
        rough = clamp(mix(rough, 0.30, 0.32) + u.style_weights_b.w, 0.08, 1.0);
    }
    glow = clamp(glow * u.style_weights_a.w, 0.0, 1.0);

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
    let steep = smoothstep(0.76, 0.30, abs(n_base.y));
    let bridge_band = smoothstep(3.2, 4.4, p.y) * smoothstep(9.6 * d_scale, 7.6 * d_scale, p.y);
    let sat_rg = abs(alb_srgb.r - alb_srgb.g) + abs(alb_srgb.g - alb_srgb.b);
    let is_stone_like = smoothstep(0.28, 0.10, sat_rg) * (1.0 - is_greenish);

    // (1) Rich Triplanar Geological Rock/Cliff Stratification + Carved Nordic Ashlar Masonry
    if (steep > 0.04) {
        let strata_warp = vnoise2(p.xz * 0.55) * 2.8 + macro_n * 1.4;
        let strata_lo = sin(p.y * 3.8 + strata_warp);
        let strata_hi = sin(p.y * 9.6 + strata_warp * 1.7) * 0.45;
        let strata = (strata_lo * 0.65 + strata_hi * 0.35) * 0.16;

        // Carved Nordic ashlar stone-block courses on bridge arches & stone walls
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

    // (1B) Carved Nordic Ashlar Flagstone Paving on Bridge Spans (top deck)
    let under_open = 1.0 - sample_density(p - vec3<f32>(0.0, 2.2, 0.0)).r;
    let is_bridge_deck = bridge_band * smoothstep(0.68, 0.88, n_base.y) * smoothstep(0.25, 0.72, under_open) * (1.0 - is_greenish);
    if (is_bridge_deck > 0.05) {
        let bx = abs(fract(p.x * 0.55 + floor(p.z * 0.55) * 0.5) - 0.5);
        let bz = abs(fract(p.z * 0.55) - 0.5);
        let flag_mortar = smoothstep(0.43, 0.495, max(bx, bz));
        let block_var = (hash2(floor(p.xz * 0.55)) - 0.5) * 0.14;
        let deck_tint = vec3<f32>(0.54, 0.58, 0.66) * (0.94 + block_var);
        alb_srgb = mix(alb_srgb, deck_tint * (1.0 - flag_mortar * 0.28), is_bridge_deck * 0.58);
        n = normalize(n + vec3<f32>((bx - 0.25) * flag_mortar * 0.25, 0.0, (bz - 0.25) * flag_mortar * 0.25) * is_bridge_deck);
    }

    // (2) Warm weathered timber plank seams & wood grain on horizontal bridge decks
    let wood_deck = is_warm_wood * smoothstep(0.55, 0.82, n_base.y);
    if (wood_deck > 0.05) {
        let grain = sin(p.z * 11.0 + vnoise2(p.xz * 2.8) * 3.2) * 0.09;
        let plank_seam = smoothstep(0.90, 0.99, abs(sin(p.x * 2.4)));
        alb_srgb = clamp(alb_srgb * (1.0 + (grain - plank_seam * 0.22) * wood_deck), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // (3) Snow-Dusted Upward-Facing Alpine Ledges (`N.y > 0.72`) + Crystalline Sparkle!
    let overhead_rock = sample_density(p + vec3<f32>(0.0, 2.6, 0.0)).r
                      + sample_density(p + vec3<f32>(0.0, 4.8, 0.0)).r;
    let exposed_sky = 1.0 - smoothstep(0.30, 0.90, overhead_rock);
    if (style_id == 0) {
        let snow_ledge = smoothstep(0.72, 0.88, n_base.y)
                       * smoothstep(0.46, 0.72, alt_norm + (macro_n - 0.5) * 0.22)
                       * exposed_sky
                       * (1.0 - is_bridge_deck * 0.92)
                       * (1.0 - wood_deck * 0.92)
                       * (1.0 - smoothstep(0.16, 0.40, glow));
        let high_snow = smoothstep(0.62, 0.82, luma_s) * smoothstep(0.46, 0.64, alt_norm) * smoothstep(0.55, 0.82, n_base.y);
        let snow_w = clamp(max(snow_ledge * 0.64, high_snow * 0.74), 0.0, 1.0);
        if (snow_w > 0.04) {
            let speck = hash2(floor(p.xz * 7.2) + floor(p.y * 4.5));
            let sparkle = step(0.90, speck) * 0.24 * snow_w;
            let snow_col = vec3<f32>(0.78, 0.87, 0.97) + vec3<f32>(sparkle);
            alb_srgb = mix(alb_srgb, clamp(snow_col, vec3<f32>(0.0), vec3<f32>(1.0)), snow_w * 0.58);
            rough = mix(rough, 0.28, snow_w * 0.45);
        }
    } else {
        let snow_w = smoothstep(0.58, 0.78, luma_s) * smoothstep(0.48, 0.62, alt_norm) * smoothstep(0.40, 0.80, n_base.y);
        if (snow_w > 0.05) {
            let speck = hash2(floor(p.xz * 6.5) + floor(p.y * 4.0));
            let sparkle = step(0.91, speck) * 0.24 * snow_w;
            alb_srgb = clamp(alb_srgb + vec3<f32>(sparkle), vec3<f32>(0.0), vec3<f32>(1.0));
            rough = mix(rough, 0.28, snow_w * 0.45);
        }
    }

    // (4) Wet shoreline pebble darkening on low-roughness wet stone/banks
    let wet_w = smoothstep(0.50, 0.22, rough);
    if (wet_w > 0.05) {
        let pebble = (fine_n - 0.5) * 0.14;
        alb_srgb = clamp(alb_srgb * (0.94 + pebble * wet_w), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // (5) Incandescent Molten Fissure Glow (`thermal > 0.55`) & Volcanic Lava Veins
    var magma_emit = vec3<f32>(0.0);
    if (raw_thermal > 0.45 || glow > 0.05 || (style_id == 1 && alt_norm < 0.42)) {
        let crack_n = triplanar_fbm(p + vec3<f32>(0.0, -sim_t * 0.16, 0.0), n_base, 1.55);
        let crack_sharp = 1.0 - smoothstep(0.015, 0.085, abs(crack_n - 0.50));
        let pulse = 0.82 + 0.18 * sin(sim_t * 3.4 + p.x * 0.5 + p.z * 0.5);
        let hot_fissure = smoothstep(0.55, 0.74, raw_thermal);
        let crack_boost = select(0.0, crack_sharp * 0.42 * u.style_weights_c.w * smoothstep(0.42, 0.12, alt_norm), style_id == 1);
        if (hot_fissure > 0.02) {
            let char_rim = smoothstep(0.14, 0.06, abs(crack_n - 0.50)) * (1.0 - crack_sharp);
            alb_srgb = mix(alb_srgb, vec3<f32>(0.09, 0.06, 0.08), char_rim * hot_fissure * 0.65);
            magma_emit = blackbody_ramp(clamp(0.55 + crack_sharp * 0.45, 0.0, 1.0), vec3<f32>(1.0, 0.48, 0.08), style_id)
                       * crack_sharp * hot_fissure * pulse * 1.45;
        }
        glow = clamp(glow + crack_boost + glow * crack_sharp * 0.40, 0.0, 1.0);
    }

    // (6) Arcane Thermal holographic contour scanlines & subtle spatial grid (Style 2)
    if (style_id == 2) {
        let contour = smoothstep(0.91, 0.99, abs(sin(p.y * 3.14159)));
        let grid_line = smoothstep(0.96, 0.995, max(abs(sin(p.x * 0.7854)), abs(sin(p.z * 0.7854))));
        alb_srgb = clamp(alb_srgb + vec3<f32>(0.05, 0.30, 0.40) * (contour * 0.60 + grid_line * 0.22), vec3<f32>(0.0), vec3<f32>(1.0));
        glow = max(glow, (contour * 0.22 + grid_line * 0.08) * u.style_weights_c.w);
    }

    // (7) Emerald Cavern bioluminescent spore veins & jade crystal pulse (Style 3)
    if (style_id == 3) {
        let spore_n = triplanar_fbm(p + vec3<f32>(sim_t * 0.08, 0.0, -sim_t * 0.06), n_base, 1.15);
        let spore_vein = 1.0 - smoothstep(0.42, 0.50, abs(spore_n - 0.5));
        let pulse = 0.75 + 0.25 * sin(sim_t * 1.8 + p.x * 0.3 + p.z * 0.3);
        let cavern_band = smoothstep(10.5, 3.2, p.y) * (0.45 + 0.55 * steep);
        alb_srgb = clamp(alb_srgb + vec3<f32>(0.04, 0.34, 0.22) * spore_vein * cavern_band * 0.65, vec3<f32>(0.0), vec3<f32>(1.0));
        glow = max(glow, spore_vein * cavern_band * 0.28 * pulse * u.style_weights_c.w);
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
    var ambient_bot = vec3<f32>(0.04, 0.06, 0.12);
    if (style_id == 1) {
        ambient_bot = vec3<f32>(0.14, 0.05, 0.03);
    } else if (style_id == 2) {
        ambient_bot = vec3<f32>(0.04, 0.08, 0.20);
    } else if (style_id == 3) {
        ambient_bot = vec3<f32>(0.03, 0.14, 0.10);
    }
    let ambient = mix(ambient_bot, u.sky_top.rgb * 0.85 + u.sky_horizon.rgb * 0.34, sky_w) * ao;
    let direct = u.sun_col.rgb * light_scalar;
    let bounce_emit = compute_emissive_bounce_3d(p, n_base, style_id);

    // Subterranean vaulted cavern lighting + Cool Cyan/Amethyst Cavern Crystal Veins!
    let in_cave = smoothstep(0.38, 1.12, overhead_rock) * smoothstep(7.2 * d_scale, 3.2 * d_scale, p.y);
    var cave_lit = vec3<f32>(0.0);
    if (in_cave > 0.02) {
        let cave_albedo = max(albedo, vec3<f32>(0.085, 0.095, 0.125));
        let torch_dir = normalize(u.cam_pos.xyz - p + vec3<f32>(0.0, 0.8, 0.0));
        let torch_att = 1.0 / (1.0 + hit_t * hit_t * 0.0022);
        let torch_nd = max(dot(n, torch_dir), 0.15);
        var cave_tint = vec3<f32>(0.92, 0.42, 0.16);
        var cave_rim_col = vec3<f32>(0.22, 0.36, 0.62);
        if (style_id == 2) {
            cave_tint = vec3<f32>(0.34, 0.56, 0.96);
            cave_rim_col = vec3<f32>(0.62, 0.24, 0.78);
        } else if (style_id == 3) {
            cave_tint = vec3<f32>(0.18, 0.86, 0.52);
            cave_rim_col = vec3<f32>(0.12, 0.58, 0.62);
        }
        let magma_glow_amb = cave_tint * (0.28 + 0.52 * max(-n.y, 0.0) + 0.35 * torch_nd * torch_att);
        let cavern_rim = cave_rim_col * (0.22 + 0.65 * pow(1.0 - ndv, 2.0));

        // Faceted glowing cyan/turquoise & amethyst subterranean mineral crystal veins
        let cryst_n = triplanar_fbm(p * 0.85, n_base, 1.25);
        let cryst_vein = 1.0 - smoothstep(0.015, 0.075, abs(cryst_n - 0.52));
        let hue_sel = smoothstep(0.35, 0.65, triplanar_noise(p, n_base, 0.28));
        let cyan_col = vec3<f32>(0.08, 0.90, 0.94);
        let amet_col = vec3<f32>(0.68, 0.24, 1.00);
        let cryst_col = mix(cyan_col, amet_col, hue_sel);
        let cryst_pulse = 0.74 + 0.26 * sin(sim_t * 2.2 + p.x * 0.35 + p.z * 0.42);
        let cryst_glint = pow(ndh, 28.0) * 0.80 + fresnel * 0.65;
        let crystal_emit = cryst_col * cryst_vein * (0.55 + cryst_glint) * cryst_pulse;

        cave_lit = (cave_albedo * (magma_glow_amb + cavern_rim + bounce_emit * 2.2) + crystal_emit * 1.15) * in_cave;
    }

    var hdr = albedo * (direct + ambient * 0.80 + bounce_emit) + cave_lit + magma_emit + u.sun_col.rgb * spec;

    // Signature stylized shadow/highlight ramp when Pixel-2xSS mode is active
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
            cool_ink = srgb_to_linear(vec3<f32>(0.04, 0.16, 0.14));
            warm_hi = srgb_to_linear(vec3<f32>(0.76, 1.00, 0.84));
        }
        let shadow_t = clamp(1.0 - light_scalar * 1.45, 0.0, 1.0) * (1.0 - clamp(glow * 2.0, 0.0, 1.0));
        hdr = mix(hdr, cool_ink * (0.25 + 0.55 * ao), shadow_t * 0.18);
        let hi_t = clamp((light_scalar - 0.70) * 1.8, 0.0, 1.0);
        hdr = mix(hdr, hdr * 0.83 + warm_hi * 0.23, hi_t * 0.25);
    }

    if (glow > 0.04) {
        hdr += blackbody_ramp(glow, albedo, style_id);
    }

    // Soft glowing rock cross-section rim along Smart 3D Cave/Bridge Ceiling Cutaway edge
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

    let fog_t = clamp((hit_t - 55.0) / 195.0, 0.0, 0.76);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t * (1.0 - clamp(glow, 0.0, 0.85)));

    return vec4<f32>(hdr, 1.0);
}
