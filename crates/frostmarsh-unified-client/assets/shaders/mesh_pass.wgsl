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
    flags: vec4<f32>,           // x = mat_override (<0 keep vertex), y = selective_tint (>0.5 only mat 12), z = roll (rad), w = reserved
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

    let cr = cos(inst.flags.z);
    let sr = sin(inst.flags.z);
    let p_rolled = vec3<f32>(local_p.x * cr - local_p.y * sr, local_p.x * sr + local_p.y * cr, local_p.z);
    let n_rolled = vec3<f32>(normal.x * cr - normal.y * sr, normal.x * sr + normal.y * cr, normal.z);

    let cp = cos(inst.scale_pitch.w);
    let sp = sin(inst.scale_pitch.w);
    let p_pitched = vec3<f32>(p_rolled.x, p_rolled.y * cp - p_rolled.z * sp, p_rolled.y * sp + p_rolled.z * cp);
    let n_pitched = vec3<f32>(n_rolled.x, n_rolled.y * cp - n_rolled.z * sp, n_rolled.y * sp + n_rolled.z * cp);

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
        let patch_l = fbm2(wp.xz * 0.18);
        let detail_n = vnoise(wp.xz * 0.95);
        let tuft = hash2(floor(wp.xz * 4.2));
        let speck = hash2(floor(wp.xz * 7.5));

        // Fast single-octave procedural normal bump for crisp ground/turf/rock relief at 70+ FPS
        let h_x = vnoise((wp.xz + vec2<f32>(0.25, 0.0)) * 0.95) - detail_n;
        let h_z = vnoise((wp.xz + vec2<f32>(0.0, 0.25)) * 0.95) - detail_n;
        n = normalize(n + vec3<f32>(-h_x * 0.42, 0.0, -h_z * 0.42));

        // Rich painterly macro + micro biome variation over compositional A+ ground albedo
        alb = clamp(
            alb * (0.84 + patch_l * 0.24 + (detail_n - 0.5) * 0.16 + (tuft - 0.5) * 0.08),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );

        // Damp dark shoreline bank & pebble sheen right above water level
        let shore_t = smoothstep(0.56, 0.14, wp.y);
        if (shore_t > 0.02) {
            let wet_bank = vec3<f32>(0.18, 0.22, 0.24) * (0.85 + detail_n * 0.30);
            alb = mix(alb, wet_bank, shore_t * 0.45);
            metallic = max(metallic, shore_t * 0.28);
        }

        // Cliff / riverbank slope rock strata on steep faces
        let slope = smoothstep(0.89, 0.62, n.y);
        if (slope > 0.02) {
            let strata = sin(wp.y * 5.8 + vnoise(wp.xz * 0.55) * 2.6) * 0.16;
            let cliff_col = clamp(
                vec3<f32>(0.30, 0.36, 0.48) * (0.94 + strata + (detail_n - 0.5) * 0.22),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            alb = mix(alb, cliff_col, slope * 0.78);
            n = normalize(n + vec3<f32>(strata * 0.42, 0.0, -strata * 0.42));
        }

        // High-elevation alpine snow dusting on flat upper ridges
        let alpine_snow = smoothstep(3.4, 4.3, wp.y) * smoothstep(0.68, 0.90, n.y);
        if (alpine_snow > 0.02) {
            let snow_spark = step(0.93, speck) * 0.18;
            alb = mix(alb, vec3<f32>(0.84, 0.91, 0.98) + vec3<f32>(snow_spark), alpine_snow * 0.52);
            metallic = max(metallic, alpine_snow * 0.25);
        }

        // Frost / freezing shimmer when cold (temp_n < 0.075 => below freezing!)
        let frost_t = smoothstep(0.088, 0.044, temp_n);
        if (frost_t > 0.05) {
            let ice_flash = flash_pal(34);
            let crystal_sparkle = step(0.89, speck);
            let frost_col = mix(vec3<f32>(0.72, 0.90, 0.99), ice_flash.rgb, 0.30 + crystal_sparkle * 0.48);
            alb = mix(alb, frost_col, frost_t * 0.64);
            metallic = max(metallic, frost_t * 0.52);
        }

        // Live wildfire / molten ember glow with glowing lava/charcoal crack veins!
        if (burn_t > 0.04 || temp_n > 0.24) {
            let heat = max(burn_t, smoothstep(0.24, 0.62, temp_n));
            let flame_n = fbm2(wp.xz * 1.45 + vec2<f32>(t * 0.9, -t * 1.6));
            let crack = smoothstep(0.42, 0.52, abs(vnoise(wp.xz * 2.4) - 0.5));
            let ember_p = flash_pal(27);
            let core_p = flash_pal(29);
            let char_col = vec3<f32>(0.10, 0.08, 0.11);
            let ember_col = mix(ember_p.rgb, core_p.rgb, smoothstep(0.30, 0.76, flame_n * heat + heat * 0.35));
            let cracked_burn = mix(ember_col, char_col, (1.0 - crack) * 0.42 * (1.0 - heat * 0.5));
            let mix_f = clamp(heat * (0.58 + flame_n * 0.62), 0.0, 1.0);
            alb = mix(alb, cracked_burn, mix_f);
            glow = max(glow, mix_f * (0.72 + crack * 0.40) * max(ember_p.a, core_p.a));
        }
    } else if (mat_id == 3) {
        // === MATERIAL 3: LIVE FLOWING RIVER & WATERFALL (caustics + Fresnel sky reflection + foam!) ===
        let fdir = select(vec2<f32>(0.0, 1.0), normalize(flow_v), length(flow_v) > 0.05);
        let speed = 0.95 + clamp(length(flow_v), 0.0, 2.2) * 0.72;
        let is_waterfall = abs(in.normal.y) < 0.45;
        let p_uv = select(wp.xz, vec2<f32>(wp.x + wp.z, wp.y * 1.35), is_waterfall);
        let w1 = vnoise(p_uv * 0.92 - fdir * t * 1.25 * speed);
        let w2 = vnoise(p_uv * 1.85 - fdir * t * 0.92 * speed + vec2<f32>(7.3, -4.1));
        let w3 = vnoise(p_uv * 3.60 - fdir * t * 1.55 * speed + vec2<f32>(-3.2, 5.7));
        let ripple = w1 * 0.45 + w2 * 0.35 + w3 * 0.20;
        let caustic = pow(clamp(1.0 - abs(w1 - w2) * 2.8, 0.0, 1.0), 3.0);
        if (!is_waterfall) {
            n = normalize(vec3<f32>((w1 - 0.5) * 0.42 + (w3 - 0.5) * 0.18, 1.0, (w2 - 0.5) * 0.42));
        }

        let deep_w = flash_pal(31).rgb * 0.88;
        let mid_w = flash_pal(32).rgb;
        let sunlit_w = flash_pal(33).rgb;
        let crest_w = flash_pal(34).rgb;
        alb = mix(deep_w, mid_w, smoothstep(0.22, 0.66, ripple));
        alb = mix(alb, sunlit_w, smoothstep(0.56, 0.78, ripple) * 0.55 + caustic * 0.28);
        let crest = smoothstep(0.72, 0.84, ripple);
        alb = mix(alb, mix(crest_w, vec3<f32>(0.90, 0.98, 1.0), 0.58), crest * 0.65);

        if (is_waterfall) {
            let streak = smoothstep(0.35, 0.78, w2 * 0.6 + w3 * 0.4);
            alb = mix(vec3<f32>(0.42, 0.80, 0.98), vec3<f32>(0.90, 0.98, 1.0), streak * 0.72);
            glow = max(glow, 0.16);
        }

        // Subtle steam glow when water is heated
        if (temp_n > 0.24) {
            let boil = smoothstep(0.24, 0.55, temp_n);
            alb = mix(alb, vec3<f32>(0.78, 0.90, 0.98), boil * 0.35);
        }
    } else if (mat_id == 6) {
        // === MATERIAL 6: CARVED ICE RAFT / GLACIAL BRIDGE (subsurface cyan + crystalline facets) ===
        let ice_p = flash_pal(35);
        let sp = hash2(floor(wp.xz * 4.8));
        let facet = vnoise(wp.xz * 2.2);
        let crack = smoothstep(0.46, 0.50, abs(facet - 0.5));
        alb = mix(alb, vec3<f32>(0.44, 0.84, 0.98), 0.18 * (1.0 - facet));
        alb = mix(alb, ice_p.rgb, 0.24 + step(0.87, sp) * 0.36);
        alb *= 1.0 - crack * 0.15;
        n = normalize(n + vec3<f32>((facet - 0.5) * 0.22, 0.0, (sp - 0.5) * 0.14));
        metallic = 0.74;
    } else if (mat_id == 5) {
        // Emissive crystal / glowing relic / fire core / dash halo
        let pulse = flash_pal(35).a;
        glow = max(glow, 0.85 * (0.85 + 0.15 * pulse));
    } else if (mat_id == 1) {
        // Rock / stone pillar masonry & chiseled bevel
        let strata = sin(wp.y * 4.8 + vnoise(wp.xz * 0.8) * 2.2) * 0.12;
        let grit = (vnoise(wp.xz * 3.4) - 0.5) * 0.18;
        alb = clamp(alb * (0.98 + strata + grit), vec3<f32>(0.0), vec3<f32>(1.0));
        n = normalize(n + vec3<f32>(strata * 0.24, 0.0, grit * 0.24));
    } else if (mat_id == 4) {
        // Wood bark / weathered bridge planks
        let grain = sin((wp.x + wp.z) * 14.0 + vnoise(wp.xz * 3.8) * 2.4) * 0.09;
        let plank_seam = smoothstep(0.92, 0.99, abs(sin(wp.x * 3.14159)));
        alb = clamp(alb * (1.0 + grain - plank_seam * 0.18), vec3<f32>(0.0), vec3<f32>(1.0));
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
    fill += u.bounce_col.rgb * horiz * (0.26 + 0.54 * away);

    let ndl = max(dot(n, u.sun_dir.xyz), 0.0) * sh;
    var light_scalar = ndl + 0.08 * max(dot(n, view_dir), 0.0);
    if (u.viewport.z > 0.5) {
        // Smooth-bevel stylized cel ramp (55% stepped + 45% continuous lighting for rich form!)
        let q_l = round(light_scalar * u.viewport.z) / u.viewport.z;
        light_scalar = mix(light_scalar, q_l, 0.55);
    }

    var col = alb * (fill + u.sun_col.rgb * light_scalar) * u.sun_dir.w;

    // Subtle Cool-Indigo shadow pull + Warm-Cream highlight push
    let cool_ink = vec3<f32>(0.149, 0.118, 0.282);
    let warm_hi = vec3<f32>(1.000, 0.957, 0.839);
    let shadow_t = clamp(1.0 - light_scalar * 1.45, 0.0, 1.0);
    col = mix(col, cool_ink, shadow_t * 0.17);
    let hi_t = clamp((light_scalar - 0.78) * 1.8, 0.0, 1.0);
    col = mix(col, warm_hi, hi_t * 0.13);

    let ndv = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - ndv, 3.2);

    // Metallic armor / gold / ice specular glint + rim
    if (metallic > 0.2) {
        let hv = normalize(u.sun_dir.xyz + view_dir);
        let spec = pow(max(dot(n, hv), 0.0), 28.0) * sh;
        col += (u.sun_col.rgb * spec * 0.74 + u.sky_col.rgb * fresnel * 0.48) * metallic;
    } else if (mat_id != 9 && mat_id != 3) {
        // Crisp hero & prop readability rim light against dark terrain / night scenes
        col += (u.sky_col.rgb * 0.55 + u.sun_col.rgb * 0.35) * fresnel * 0.28;
    }

    // Water Fresnel sky reflection + crisp sun sparkle (keeps rich Nordic fjord azure!)
    if (mat_id == 3) {
        let hv = normalize(u.sun_dir.xyz + view_dir);
        let spec = pow(max(dot(n, hv), 0.0), 72.0) * sh;
        let sky_refl = mix(vec3<f32>(0.14, 0.44, 0.72), u.sky_col.rgb * 1.35, 0.48);
        col = mix(col, sky_refl, clamp(fresnel * 0.34, 0.0, 0.28));
        col += u.sun_col.rgb * spec * 1.30;
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
                col += (alb * 0.85 + vec3<f32>(0.10)) * u.point_col[i].rgb * att * att * (0.38 + 0.62 * l_ndl);
            }
        }
    }

    if (glow > 0.05) {
        col = mix(col, alb * 1.50, clamp(glow, 0.0, 1.0));
    }

    // Directional sun-scattered atmospheric perspective fog
    let fog_t = 0.62 * smoothstep(u.sky_col.w, u.bounce_col.w, dist);
    let sun_scatter = pow(max(dot(-view_dir, u.sun_dir.xyz), 0.0), 5.0);
    let atmo_fog = u.fog_col.rgb + u.sun_col.rgb * sun_scatter * 0.26;
    col = mix(col, atmo_fog, fog_t * (1.0 - clamp(glow, 0.0, 1.0) * 0.85));
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
