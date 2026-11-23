// Approach 4 Primary Pass for Live Neural Frostmarsh
// - 2048x2048 4-tap PCF directional shadows + hemisphere bounce
// - 9-step Cool-Indigo (#261e48) / Warm-Cream (#fff4d6) Hue-Shifted Tone Ramp
// - Live 1 KB Animated `flash()` Palette Texture (256x1 RGBA8) for wildfires, embers, frost & flowing river
// - Dynamic 8-slot Firelight / Spell / Relic Point Lights
// - Compositional A+ terrain vertex shading (temperature, flow direction, burning, freezing, oil sheen, ice shimmer)

struct SceneUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,
    cam_fwd: vec4<f32>,
    origin_time: vec4<f32>,     // xyz = floating origin, w = sim_time
    viewport: vec4<f32>,        // xy = ss_res, z = cel_bands (9.0), w = shadow_bias
    sun_dir: vec4<f32>,         // xyz = normalized dir to sun, w = exposure
    sun_col: vec4<f32>,         // rgb = sun color, w = shadow_strength
    sky_col: vec4<f32>,         // rgb = sky fill, w = fog_start
    bounce_col: vec4<f32>,      // rgb = bounce fill, w = fog_end
    fog_col: vec4<f32>,         // rgb = haze/bg color, w = unused
    point_pos: array<vec4<f32>, 8>,
    point_col: array<vec4<f32>, 8>,
};

struct MeshInstance {
    pos_yaw: vec4<f32>,         // camera-relative xyz, w = yaw
    scale_pitch: vec4<f32>,     // xyz = scale, w = pitch
    tint_glow: vec4<f32>,       // rgb = tint multiplier, w = extra glow
    flags: vec4<f32>,           // x = mat_override (<0 keep vertex), y = selective_tint (>0.5 only mat 12)
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<storage, read> instances: array<MeshInstance>;
@group(0) @binding(2) var shadow_tex: texture_depth_2d;
@group(0) @binding(3) var shadow_samp: sampler_comparison;
@group(0) @binding(4) var flash_pal_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) mat_glow: vec2<f32>, // x = mat_id, y = inst glow
    @location(4) extra: vec4<f32>, // x = temp_norm [0..1], y = burn_intensity [0..1], z = flow_x, w = flow_z
};

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
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
    return vnoise(p) * 0.58
         + vnoise(p * 2.13 + vec2<f32>(13.7, -9.2)) * 0.28
         + vnoise(p * 4.41 - vec2<f32>(5.1, 19.3)) * 0.14;
}

fn flash_pal(idx: i32) -> vec4<f32> {
    let c = clamp(idx, 0, 255);
    return textureLoad(flash_pal_tex, vec2<i32>(c, 0), 0);
}

fn sample_shadow(rel_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    let biased = rel_pos + n * 0.14 + u.sun_dir.xyz * 0.10;
    let lclip = u.light_vp * vec4<f32>(biased, 1.0);
    let ndc = lclip.xyz / lclip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let texel = 1.0 / 2048.0;
    let z = ndc.z - u.viewport.w;
    var s = 0.0;
    s += textureSampleCompare(shadow_tex, shadow_samp, uv + vec2<f32>(-texel, -texel), z);
    s += textureSampleCompare(shadow_tex, shadow_samp, uv + vec2<f32>( texel, -texel), z);
    s += textureSampleCompare(shadow_tex, shadow_samp, uv + vec2<f32>(-texel,  texel), z);
    s += textureSampleCompare(shadow_tex, shadow_samp, uv + vec2<f32>( texel,  texel), z);
    return mix(1.0, s * 0.25, u.sun_col.w);
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color_mat: vec4<f32>,
    @location(3) extra: vec4<f32>,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let inst = instances[iid];
    let local_p = pos * inst.scale_pitch.xyz;

    let cp = cos(inst.scale_pitch.w);
    let sp = sin(inst.scale_pitch.w);
    let p_pitched = vec3<f32>(local_p.x, local_p.y * cp - local_p.z * sp, local_p.y * sp + local_p.z * cp);
    let n_pitched = vec3<f32>(normal.x, normal.y * cp - normal.z * sp, normal.y * sp + normal.z * cp);

    let cy = cos(inst.pos_yaw.w);
    let sy = sin(inst.pos_yaw.w);
    let rel_p = inst.pos_yaw.xyz + vec3<f32>(
        p_pitched.x * cy - p_pitched.z * sy,
        p_pitched.y,
        p_pitched.x * sy + p_pitched.z * cy,
    );
    let world_n = normalize(vec3<f32>(
        n_pitched.x * cy - n_pitched.z * sy,
        n_pitched.y,
        n_pitched.x * sy + n_pitched.z * cy,
    ));

    let mat_id = select(color_mat.w, inst.flags.x, inst.flags.x >= 0.0);
    // Selective tint mode (flags.y > 0.5): ONLY mat 12 (faction cloth) receives tint_glow.rgb
    let use_tint = (inst.flags.y < 0.5) || (abs(mat_id - 12.0) < 0.5);
    let final_col = select(color_mat.rgb, color_mat.rgb * inst.tint_glow.rgb, use_tint);

    var out: VsOut;
    out.clip_pos = u.vp * vec4<f32>(rel_p, 1.0);
    out.rel_pos = rel_p;
    out.normal = world_n;
    out.color = final_col;
    out.mat_glow = vec2<f32>(mat_id, inst.tint_glow.w);
    out.extra = extra;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    var alb = in.color;
    var glow = in.mat_glow.y;
    var metallic = 0.0;
    let mat_id = i32(round(in.mat_glow.x));
    let wp = in.rel_pos + u.origin_time.xyz;
    let t = u.origin_time.w;

    let temp_n = in.extra.x;
    let burn_t = in.extra.y;
    let flow_v = in.extra.zw;

    // === MATERIAL 9: LIVE NEURAL A+ COMPOSITIONAL TERRAIN ===
    if (mat_id == 9) {
        let patch_l = fbm2(wp.xz * 0.22);
        let tuft = hash2(floor(wp.xz * 3.8));
        let speck = hash2(floor(wp.xz * 6.0));

        // Organic painterly micro-variation over compositional A+ ground albedo
        alb = clamp(alb * (0.88 + patch_l * 0.24 + (tuft - 0.5) * 0.11), vec3<f32>(0.0), vec3<f32>(1.0));

        // Cliff / riverbank slope rock strata on steep faces
        let slope = smoothstep(0.88, 0.66, n.y);
        if (slope > 0.02) {
            let strata = sin(wp.y * 5.2 + vnoise(wp.xz * 0.6) * 2.2) * 0.14;
            let cliff_col = clamp(vec3<f32>(0.34, 0.40, 0.52) * (1.0 + strata + (tuft - 0.5) * 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
            alb = mix(alb, cliff_col, slope * 0.72);
            n = normalize(n + vec3<f32>(strata * 0.38, 0.0, -strata * 0.38));
        }

        // Frost / freezing shimmer when cold (temp_n < 0.075 => below freezing!)
        let frost_t = smoothstep(0.085, 0.045, temp_n);
        if (frost_t > 0.05) {
            let ice_flash = flash_pal(34);
            let crystal_sparkle = step(0.91, speck);
            let frost_col = mix(vec3<f32>(0.68, 0.88, 0.98), ice_flash.rgb, 0.28 + crystal_sparkle * 0.45);
            alb = mix(alb, frost_col, frost_t * 0.58);
            metallic = max(metallic, frost_t * 0.45);
        }

        // Live wildfire / molten ember glow driven by 1 KB animated flash() palette!
        if (burn_t > 0.04 || temp_n > 0.24) {
            let heat = max(burn_t, smoothstep(0.24, 0.62, temp_n));
            let flame_n = fbm2(wp.xz * 1.45 + vec2<f32>(t * 0.9, -t * 1.6));
            let ember_p = flash_pal(27); // animated forge orange
            let core_p = flash_pal(29);  // animated incandescent fire core
            let ember_col = mix(ember_p.rgb, core_p.rgb, smoothstep(0.32, 0.78, flame_n * heat + heat * 0.35));
            let mix_f = clamp(heat * (0.55 + flame_n * 0.65), 0.0, 1.0);
            alb = mix(alb, ember_col, mix_f);
            glow = max(glow, mix_f * max(ember_p.a, core_p.a));
        }
    } else if (mat_id == 3) {
        // === MATERIAL 3: LIVE FLOWING RIVER WATER (uses A+ flow vector + 1 KB flash() palette!) ===
        let fdir = select(vec2<f32>(0.0, 1.0), normalize(flow_v), length(flow_v) > 0.05);
        let speed = 0.9 + clamp(length(flow_v), 0.0, 1.6) * 0.65;
        let p_uv = wp.xz;
        let w1 = vnoise(p_uv * 0.85 - fdir * t * 1.15 * speed);
        let w2 = vnoise(p_uv * 1.65 - fdir * t * 0.85 * speed + vec2<f32>(7.3, -4.1));
        let w3 = vnoise(floor(p_uv * 3.5) * 0.35 - fdir * t * 0.65);
        let ripple = w1 * 0.45 + w2 * 0.35 + w3 * 0.20;
        n = normalize(vec3<f32>((w1 - 0.5) * 0.36, 1.0, (w2 - 0.5) * 0.36));

        let deep_w = flash_pal(31).rgb;
        let mid_w = flash_pal(32).rgb;
        let sunlit_w = flash_pal(33).rgb;
        let crest_w = flash_pal(34).rgb;
        alb = mix(deep_w, mid_w, smoothstep(0.24, 0.65, ripple));
        alb = mix(alb, sunlit_w, smoothstep(0.58, 0.78, ripple) * 0.55);
        let crest = step(0.76, ripple);
        alb = mix(alb, mix(crest_w, vec3<f32>(0.88, 0.97, 1.0), 0.50), crest * 0.55);

        // Subtle steam glow when water is heated
        if (temp_n > 0.24) {
            let boil = smoothstep(0.24, 0.55, temp_n);
            alb = mix(alb, vec3<f32>(0.76, 0.88, 0.96), boil * 0.32);
        }
    } else if (mat_id == 6) {
        // === MATERIAL 6: CARVED ICE RAFT / GLACIAL BRIDGE ===
        let ice_p = flash_pal(35);
        let sp = hash2(floor(wp.xz * 4.2));
        let crack = smoothstep(0.47, 0.50, abs(vnoise(wp.xz * 1.9) - 0.5));
        alb = mix(alb, ice_p.rgb, 0.22 + step(0.88, sp) * 0.32);
        alb *= 1.0 - crack * 0.12;
        metallic = 0.65;
    } else if (mat_id == 5) {
        // Emissive crystal / glowing relic / fire core / dash halo
        let pulse = flash_pal(35).a;
        glow = max(glow, 0.82 * (0.85 + 0.15 * pulse));
    } else if (mat_id == 1) {
        // Rock / stone pillar masonry & bevel
        let strata = sin(wp.y * 4.4 + vnoise(wp.xz * 0.8) * 2.0) * 0.10;
        let grit = (hash2(floor(wp.xz * 3.0)) - 0.5) * 0.14;
        alb = clamp(alb * (1.0 + strata + grit), vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (mat_id == 4) {
        // Wood bark / planks
        let grain = sin((wp.x + wp.z) * 12.0 + vnoise(wp.xz * 3.5) * 2.2) * 0.08;
        alb = clamp(alb * (1.0 + grain), vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (mat_id == 10) {
        // Polished steel / royal gold armor & relic metal
        metallic = 1.0;
    }

    let to_cam = u.cam_pos.xyz - in.rel_pos;
    let dist = max(length(to_cam), 0.001);
    let view_dir = to_cam / dist;

    let sh = sample_shadow(in.rel_pos, n);
    var fill = mix(u.bounce_col.rgb, u.sky_col.rgb, n.y * 0.5 + 0.5);
    let horiz = 1.0 - n.y * n.y;
    let away = max(0.0, -dot(n.xz, normalize(u.sun_dir.xz + vec2<f32>(0.0001, 0.0))));
    fill += u.bounce_col.rgb * horiz * (0.24 + 0.52 * away);

    let ndl = max(dot(n, u.sun_dir.xyz), 0.0) * sh;
    var light_scalar = ndl + 0.08 * max(dot(n, view_dir), 0.0);
    if (u.viewport.z > 0.5) {
        // 9-step pixel-art tone ramp (72% discrete steps + 28% micro-bevel fidelity!)
        let q_l = round(light_scalar * u.viewport.z) / u.viewport.z;
        light_scalar = mix(light_scalar, q_l, 0.72);
    }

    var col = alb * (fill + u.sun_col.rgb * light_scalar) * u.sun_dir.w;

    // 9-step Cool-Indigo (#261e48) shadow pull + Warm-Cream (#fff4d6) highlight push!
    let cool_ink = vec3<f32>(0.149, 0.118, 0.282);
    let warm_hi = vec3<f32>(1.000, 0.957, 0.839);
    let shadow_t = clamp(1.0 - light_scalar * 1.45, 0.0, 1.0);
    col = mix(col, cool_ink, shadow_t * 0.20);
    let hi_t = clamp((light_scalar - 0.78) * 1.8, 0.0, 1.0);
    col = mix(col, warm_hi, hi_t * 0.14);

    // Metallic armor / gold / ice specular glint + rim
    if (metallic > 0.2) {
        let hv = normalize(u.sun_dir.xyz + view_dir);
        let spec = pow(max(dot(n, hv), 0.0), 24.0) * sh;
        let rim = pow(1.0 - max(dot(n, view_dir), 0.0), 2.5);
        col += (u.sun_col.rgb * spec * 0.68 + u.sky_col.rgb * rim * 0.44) * metallic;
    }

    // Water sun glint
    if (mat_id == 3) {
        let hv = normalize(u.sun_dir.xyz + view_dir);
        let spec = pow(max(dot(n, hv), 0.0), 64.0) * sh;
        col += u.sun_col.rgb * spec * 1.10;
    }

    // Up to 8 local dynamic point lights (live wildfires, fire/frost bolts, golden relic!)
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lp = u.point_pos[i];
        if (lp.w > 0.0) {
            let d3 = lp.xyz - in.rel_pos;
            let r = length(d3);
            if (r < lp.w) {
                let att = max(0.0, 1.0 - r / lp.w);
                let l_ndl = max(dot(n, d3 / max(r, 0.001)), 0.0);
                col += (alb * 0.85 + vec3<f32>(0.10)) * u.point_col[i].rgb * att * att * (0.36 + 0.64 * l_ndl);
            }
        }
    }

    if (glow > 0.05) {
        col = mix(col, alb * 1.48, clamp(glow, 0.0, 1.0));
    }

    let fog_t = 0.58 * smoothstep(u.sky_col.w, u.bounce_col.w, dist);
    col = mix(col, u.fog_col.rgb, fog_t * (1.0 - clamp(glow, 0.0, 1.0) * 0.85));
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
