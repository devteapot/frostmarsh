// Iteration 9A Renderer B Pass 1: Smooth Surface-Nets 3D Isosurface Mesh Shader
// Upgraded with Signature Frostmarsh Stylization:
//   - Triplanar painterly surface micro-detail driven purely from GPU-decoded neural 3D textures:
//       * Stone masonry bevels & geological strata on steep rock/bridge faces
//       * Alpine snow crystalline sparkle on bright upper ledges
//       * Wet shoreline pebble darkening & glossy Fresnel sheen on low-roughness surfaces
//       * Weathered timber plank seams & wood grain on warm horizontal bridge decks
//       * Incandescent molten ember cracks & glowing veins on thermal hotspots
//   - 9-step Cool-Indigo (#261e48) / Warm-Cream (#fff4d6) Hue-Shifted Tone Ramp (T-key toggleable!)
//   - Analytical trilinear ∇solid_density normals, 3D-texture cave AO, 3D-texture raymarched sun shadows,
//     and 3D-texture local emissive bounce!

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,      // xyz = cam_pos, w = sim_time
    sun_dir: vec4<f32>,      // xyz = sun_dir, w = exposure
    sun_col: vec4<f32>,      // rgb = sun_col, w = focal_px
    sky_top: vec4<f32>,      // rgb = sky_top, w = style_id (0, 1, 2)
    sky_horizon: vec4<f32>,  // rgb = sky_horizon, w = stylized (1.0 = Pixel-2xSS, 0.0 = Clean-3D)
    vol_dims: vec4<f32>,     // xyz = dims, w = tonemap_direct
    brick_dims: vec4<f32>,   // xyz = brick_dims, w = brick_size
    viewport: vec4<f32>,     // xy = w/h, zw = inv_w/h
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

fn blackbody_ramp(temp: f32, tint: vec3<f32>) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
    let style_id = i32(round(u.sky_top.w));
    if (style_id == 2) {
        // Arcane Thermal Vision: electric cyan/magenta -> coral -> incandescent white-gold!
        let low = vec3<f32>(0.15, 0.75, 1.00);
        let mid = vec3<f32>(1.00, 0.18, 0.72);
        let hi  = vec3<f32>(1.00, 0.96, 0.78);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.4, 0.40) * (t * t * 3.8);
    }
    let low = vec3<f32>(0.95, 0.14, 0.02);
    let mid = vec3<f32>(1.00, 0.52, 0.08);
    let hi  = vec3<f32>(1.00, 0.94, 0.72);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    return mix(bb, tint * 1.35, 0.45) * (t * t * 3.8);
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
    return clamp(1.0 - occ * 0.92, 0.08, 1.0);
}

fn compute_sun_shadow_3d(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let l = u.sun_dir.xyz;
    let ndl = dot(n, l);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let dims = u.vol_dims.xyz;
    let bdims = vec3<i32>(u.brick_dims.xyz);
    let bsize = u.brick_dims.w;

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
    return vis;
}

fn compute_emissive_bounce_3d(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var bounce = vec3<f32>(0.0);
    let dirs = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 0.15, 0.0),
        vec3<f32>(-1.0, 0.15, 0.0),
        vec3<f32>(0.0, 0.15, 1.0),
        vec3<f32>(0.0, 0.15, -1.0),
        vec3<f32>(-0.7, -0.3, 0.65),
        vec3<f32>(-0.7, -0.3, -0.65)
    );
    let p_off = p + n * 0.65;
    for (var i = 0; i < 6; i = i + 1) {
        let dir = normalize(dirs[i]);
        let s1 = sample_albedo(p_off + dir * 2.8);
        let s2 = sample_albedo(p_off + dir * 7.2);
        let s3 = sample_albedo(p_off + dir * 13.5);
        let w1 = max(dot(n, dir) * 0.55 + 0.45, 0.22);
        if (s1.a > 0.06) {
            bounce += blackbody_ramp(s1.a, srgb_to_linear(s1.rgb)) * w1 * 0.48;
        }
        if (s2.a > 0.06) {
            bounce += blackbody_ramp(s2.a, srgb_to_linear(s2.rgb)) * w1 * 0.34;
        }
        if (s3.a > 0.08) {
            bounce += blackbody_ramp(s3.a, srgb_to_linear(s3.rgb)) * w1 * 0.24;
        }
    }
    let s_low = sample_albedo(vec3<f32>(p.x, max(p.y - 3.0, 1.0), p.z));
    if (s_low.a > 0.08) {
        bounce += blackbody_ramp(s_low.a, srgb_to_linear(s_low.rgb)) * 0.38;
    }
    return bounce;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let to_p = p - u.cam_pos.xyz;
    let hit_t = length(to_p);
    let rd = to_p / max(hit_t, 1e-4);

    let dm = sample_density(p);
    let at = sample_albedo(p);

    var rough = clamp(dm.a, 0.08, 1.0);
    let n_base = sample_solid_normal(p);
    var alb_srgb = clamp(at.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var glow = at.a;
    let style_id = i32(round(u.sky_top.w));
    let stylized = u.sky_horizon.w;
    let sim_t = u.cam_pos.w;

    // -------------------------------------------------------------------------
    // Triplanar Painterly Surface Micro-Detail (zero flat plastic!)
    // Driven purely from neural-decoded (albedo, roughness, glow, normal, pos)
    // -------------------------------------------------------------------------
    let macro_n = triplanar_fbm(p, n_base, 0.24);
    let detail_n = triplanar_noise(p, n_base, 1.35);
    let fine_n = triplanar_noise(p, n_base, 3.80);

    // Compute fast triplanar relief perturbation
    let d_x = triplanar_noise(p + vec3<f32>(0.22, 0.0, 0.0), n_base, 1.35) - detail_n;
    let d_y = triplanar_noise(p + vec3<f32>(0.0, 0.22, 0.0), n_base, 1.35) - detail_n;
    let d_z = triplanar_noise(p + vec3<f32>(0.0, 0.0, 0.22), n_base, 1.35) - detail_n;
    var n = normalize(n_base - vec3<f32>(d_x, d_y, d_z) * (0.28 + 0.35 * rough));

    // Painterly tonal breakup across every surface
    alb_srgb = clamp(
        alb_srgb * (0.86 + macro_n * 0.20 + (detail_n - 0.5) * 0.14 + (fine_n - 0.5) * 0.06),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let luma_s = dot(alb_srgb, vec3<f32>(0.299, 0.587, 0.114));
    let steep = smoothstep(0.76, 0.34, abs(n_base.y));

    // (1) Stone masonry bevels & geological rock strata on steep faces & cave walls
    if (steep > 0.04) {
        let strata = sin(p.y * 4.4 + vnoise2(p.xz * 0.65) * 2.4) * 0.11;
        // Chiseled cut-stone block mortar courses on mid-elevation stone arches/piers
        let course_y = abs(fract(p.y * 0.85) - 0.5);
        let row_shift = floor(p.y * 0.85) * 0.5;
        let course_x = abs(fract((p.x + p.z) * 0.42 + row_shift) - 0.5);
        let mortar = smoothstep(0.42, 0.49, max(course_y, course_x));
        let arch_band = smoothstep(3.2, 4.4, p.y) * smoothstep(9.5, 8.2, p.y);
        let stone_mod = 1.0 + strata - mortar * 0.18 * arch_band;
        alb_srgb = clamp(alb_srgb * mix(1.0, stone_mod, steep * 0.85), vec3<f32>(0.0), vec3<f32>(1.0));
        n = normalize(n + vec3<f32>(strata * 0.26, -mortar * 0.22 * arch_band, -strata * 0.26) * steep);
    }

    // (2) Warm wood grain & weathered plank seams on horizontal timber surfaces
    let warm_wood = smoothstep(0.06, 0.16, alb_srgb.r - alb_srgb.b)
                  * smoothstep(0.55, 0.82, n_base.y)
                  * smoothstep(0.65, 0.45, luma_s);
    if (warm_wood > 0.05) {
        let grain = sin(p.z * 11.0 + vnoise2(p.xz * 2.8) * 3.2) * 0.08;
        let plank_seam = smoothstep(0.90, 0.99, abs(sin(p.x * 2.4)));
        alb_srgb = clamp(alb_srgb * (1.0 + (grain - plank_seam * 0.20) * warm_wood), vec3<f32>(0.0), vec3<f32>(1.0));
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

    // (5) Molten ember cracks & glowing lava veins on hot/volcanic hotspots!
    if (glow > 0.05 || (style_id == 1 && p.y < 5.5 && steep > 0.2)) {
        let crack_n = triplanar_fbm(p + vec3<f32>(0.0, -sim_t * 0.15, 0.0), n_base, 1.45);
        let vein = smoothstep(0.43, 0.51, abs(crack_n - 0.5));
        let crack_boost = select(0.0, (1.0 - vein) * 0.22 * smoothstep(5.5, 2.2, p.y), style_id == 1);
        glow = clamp(glow + crack_boost + glow * (1.0 - vein) * 0.35, 0.0, 1.0);
    }

    // (6) Arcane Thermal contour scanlines along elevation strata in Style 2!
    if (style_id == 2) {
        let contour = smoothstep(0.92, 0.99, abs(sin(p.y * 3.14159)));
        alb_srgb = clamp(alb_srgb + vec3<f32>(0.04, 0.24, 0.32) * contour * 0.55, vec3<f32>(0.0), vec3<f32>(1.0));
        glow = max(glow, contour * 0.12);
    }

    let albedo = srgb_to_linear(alb_srgb);
    let ao = compute_ao_3d(p, n_base);
    let shadow = compute_sun_shadow_3d(p, n_base);
    let l = u.sun_dir.xyz;
    let v = -rd;
    var ndl = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);

    var light_scalar = ndl * shadow;
    if (stylized > 0.5) {
        // 9-step stylized cel bevel ramp (55% stepped + 45% smooth form preservation)
        let q_l = round(light_scalar * 9.0) / 9.0;
        light_scalar = mix(light_scalar, q_l, 0.55);
    }

    let shininess = mix(96.0, 10.0, rough);
    let spec_strength = mix(0.68, 0.04, rough);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let spec = pow(ndh, shininess) * spec_strength * (0.4 + 0.6 * fresnel) * ndl * shadow;

    let sky_w = n.y * 0.5 + 0.5;
    let ambient_bot = select(
        select(vec3<f32>(0.04, 0.06, 0.12), vec3<f32>(0.10, 0.04, 0.03), style_id == 1),
        vec3<f32>(0.03, 0.06, 0.16),
        style_id == 2
    );
    let ambient = mix(ambient_bot, u.sky_top.rgb * 0.85 + u.sky_horizon.rgb * 0.32, sky_w) * ao;
    let direct = u.sun_col.rgb * light_scalar;
    let bounce_emit = compute_emissive_bounce_3d(p, n_base);

    // Subterranean vaulted cavern lighting: warm magma bounce + cool basalt rim so cave arches,
    // pillars, ceiling ribs & glowing fissures read dramatically in linear space instead of crushing to black!
    let overhead_rock = sample_density(p + vec3<f32>(0.0, 2.8, 0.0)).r
                      + sample_density(p + vec3<f32>(0.0, 4.6, 0.0)).r;
    let in_cave = smoothstep(0.40, 1.15, overhead_rock) * smoothstep(6.2, 4.0, p.y);
    let cave_albedo = max(albedo, vec3<f32>(0.085, 0.095, 0.125));
    let torch_dir = normalize(u.cam_pos.xyz - p + vec3<f32>(0.0, 0.8, 0.0));
    let torch_att = 1.0 / (1.0 + hit_t * hit_t * 0.0022);
    let torch_nd = max(dot(n, torch_dir), 0.15);
    let magma_glow_amb = vec3<f32>(0.92, 0.42, 0.16) * (0.28 + 0.52 * max(-n.y, 0.0) + 0.35 * torch_nd * torch_att);
    let cavern_rim = vec3<f32>(0.22, 0.36, 0.62) * (0.22 + 0.65 * pow(1.0 - ndv, 2.0));
    let cave_lit = cave_albedo * (magma_glow_amb + cavern_rim + bounce_emit * 2.4) * in_cave;

    var hdr = albedo * (direct + ambient * 0.78 + bounce_emit) + cave_lit + u.sun_col.rgb * spec;

    // Signature Cool-Indigo (#261e48) shadow pull + Warm-Cream (#fff4d6) highlight push!
    if (stylized > 0.5) {
        let cool_ink = select(
            select(srgb_to_linear(vec3<f32>(0.149, 0.118, 0.282)), srgb_to_linear(vec3<f32>(0.16, 0.06, 0.10)), style_id == 1),
            srgb_to_linear(vec3<f32>(0.06, 0.08, 0.26)),
            style_id == 2
        );
        let warm_hi = select(
            select(srgb_to_linear(vec3<f32>(1.000, 0.957, 0.839)), srgb_to_linear(vec3<f32>(1.00, 0.80, 0.48)), style_id == 1),
            srgb_to_linear(vec3<f32>(0.78, 0.98, 1.00)),
            style_id == 2
        );
        let shadow_t = clamp(1.0 - light_scalar * 1.45, 0.0, 1.0) * (1.0 - clamp(glow * 2.0, 0.0, 1.0));
        hdr = mix(hdr, cool_ink * (0.25 + 0.55 * ao), shadow_t * 0.16);
        let hi_t = clamp((light_scalar - 0.72) * 1.8, 0.0, 1.0);
        hdr = mix(hdr, hdr * 0.82 + warm_hi * 0.22, hi_t * 0.24);
    }

    if (glow > 0.04) {
        hdr += blackbody_ramp(glow, albedo);
    }

    let fog_t = clamp((hit_t - 55.0) / 195.0, 0.0, 0.76);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t * (1.0 - clamp(glow, 0.0, 0.85)));

    return vec4<f32>(hdr, 1.0);
}
