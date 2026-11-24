// Pass 2: Wave 18A Living Animated Water, Waterfalls, Fresnel Reflections & Blackbody Fire Overlay (WebGPU, Z=16..32)
// Reads hardware depth from Surface-Nets + Hero Mesh pass, skips all non-fluid/non-gas bricks
// in O(bricks) via 8x8x8 brick DDA, applies animated multi-frequency wave normals, Schlick Fresnel
// sky reflections, dual-lobe crisp sun specular glints, solid-adjacency shoreline/waterfall foam,
// live 4-style fluid/lava/plasma/emerald appearance, turbulent fire/ember sparks, & ACES tonemap.

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,         // xyz = cam_pos, w = sim_time
    sun_dir: vec4<f32>,         // xyz = sun_dir, w = exposure
    sun_col: vec4<f32>,         // rgb = sun_col, w = style_mode (0..3)
    sky_top: vec4<f32>,         // rgb = sky_top, w = stylized
    sky_horizon: vec4<f32>,     // rgb = sky_horizon, w = neural_style_strength
    vol_dims: vec4<f32>,        // xyz = (W, D, H), w = 1.0 / D
    brick_dims: vec4<f32>,      // xyz = (bx, by, bz), w = brick_size
    viewport: vec4<f32>,        // xy = (w, h), zw = (1/w, 1/h)
    player_cutaway: vec4<f32>,  // xyz = capsule_pos, w = cutaway_enabled
    style_weights_a: vec4<f32>,
    style_weights_b: vec4<f32>,
    style_weights_c: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var density_tex: texture_3d<f32>;
@group(0) @binding(2) var albedo_tex: texture_3d<f32>;
@group(0) @binding(3) var brick_tex: texture_3d<f32>;
@group(0) @binding(4) var vol_samp: sampler;
@group(0) @binding(5) var scene_hdr_tex: texture_2d<f32>;
@group(0) @binding(6) var scene_depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    var out: VsOut;
    let p = pos[vid];
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
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

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
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

// Multi-frequency animated wave normals for living rivers, shorelines & plunging waterfalls
fn sample_fluid_normal(p: vec3<f32>, time: f32) -> vec3<f32> {
    let e = 0.45;
    let gx = sample_density(p + vec3<f32>(e, 0.0, 0.0)).g - sample_density(p - vec3<f32>(e, 0.0, 0.0)).g;
    let gy = sample_density(p + vec3<f32>(0.0, e, 0.0)).g - sample_density(p - vec3<f32>(0.0, e, 0.0)).g;
    let gz = sample_density(p + vec3<f32>(0.0, 0.0, e)).g - sample_density(p - vec3<f32>(0.0, 0.0, e)).g;
    var n = -vec3<f32>(gx, gy, gz);
    let len = length(n);
    if (len < 1e-4) {
        n = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        n = n / len;
    }
    let flow_z = p.z - time * 1.9;
    let w1x = sin(p.x * 0.95 + flow_z * 1.55) * 0.095
            + sin(p.x * 2.35 - p.z * 1.25 + time * 2.6) * 0.055
            + cos(p.x * 4.70 + flow_z * 3.80 - time * 3.4) * 0.032;
    let w1z = cos(p.x * 1.30 - flow_z * 1.15) * 0.095
            + cos(p.x * 0.85 + p.z * 2.65 - time * 2.9) * 0.055
            + sin(p.x * 3.90 - flow_z * 4.40 + time * 3.1) * 0.032;
    return normalize(n + vec3<f32>(w1x, 0.0, w1z));
}

// Solid-adjacency probe in `density_tex` for hugging shoreline & bridge-pier whitewater foam
fn sample_shore_adjacency(p: vec3<f32>) -> vec2<f32> {
    let sx0 = sample_density(p + vec3<f32>( 0.95, -0.25,  0.00)).r;
    let sx1 = sample_density(p + vec3<f32>(-0.95, -0.25,  0.00)).r;
    let sz0 = sample_density(p + vec3<f32>( 0.00, -0.25,  0.95)).r;
    let sz1 = sample_density(p + vec3<f32>( 0.00, -0.25, -0.95)).r;
    let bed = sample_density(p + vec3<f32>( 0.00, -1.05,  0.00)).r;
    let shore = max(max(sx0, sx1), max(sz0, sz1));
    return vec2<f32>(shore, bed);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn blackbody_ramp(temp: f32, tint: vec3<f32>, style_id: i32) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
    if (style_id == 2) {
        let low = vec3<f32>(0.12, 0.80, 1.00);
        let mid = vec3<f32>(1.00, 0.18, 0.72);
        let hi  = vec3<f32>(1.00, 0.96, 0.80);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.4, 0.35) * (t * t * 3.8);
    }
    if (style_id == 3) {
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

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let mapped = clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

fn intersect_aabb(ro: vec3<f32>, inv_rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let t0 = (bmin - ro) * inv_rd;
    let t1 = (bmax - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let depth = textureLoad(scene_depth_tex, pix, 0);

    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let p_near_h = u.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let p_far_h = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let p_near = p_near_h.xyz / p_near_h.w;
    let p_far = p_far_h.xyz / p_far_h.w;

    let ro = u.cam_pos.xyz;
    let rd = normalize(p_far - p_near);
    let style_id = i32(round(u.sun_col.w));
    let sim_t = u.cam_pos.w;

    var bg_hdr: vec3<f32>;
    var t_depth_max = 320.0;
    if (depth < 0.99999) {
        bg_hdr = textureLoad(scene_hdr_tex, pix, 0).rgb;
        let surf_h = u.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
        let surf_p = surf_h.xyz / surf_h.w;
        t_depth_max = length(surf_p - ro);
    } else {
        bg_hdr = eval_sky(rd);
    }

    let safe_rd = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let inv_rd = 1.0 / safe_rd;
    let dims = u.vol_dims.xyz;
    let bdims = vec3<i32>(u.brick_dims.xyz);
    let bsize = u.brick_dims.w;
    let hit_box = intersect_aabb(ro, inv_rd, vec3<f32>(0.01), dims - vec3<f32>(0.01));

    let t_end = min(hit_box.y, t_depth_max);
    if (hit_box.x >= t_end) {
        return vec4<f32>(aces_tonemap(bg_hdr * u.sun_dir.w), 1.0);
    }

    var t = hit_box.x + 0.002;
    let p_start = ro + rd * t;
    var bc = clamp(vec3<i32>(floor(p_start / bsize)), vec3<i32>(0), bdims - vec3<i32>(1));
    let step_i = vec3<i32>(
        select(-1, 1, rd.x >= 0.0),
        select(-1, 1, rd.y >= 0.0),
        select(-1, 1, rd.z >= 0.0)
    );
    let next_bound = (vec3<f32>(bc) + vec3<f32>(
        select(0.0, 1.0, rd.x >= 0.0),
        select(0.0, 1.0, rd.y >= 0.0),
        select(0.0, 1.0, rd.z >= 0.0)
    )) * bsize;
    var t_max_v = (next_bound - ro) * inv_rd;
    let t_delta = abs(bsize * inv_rd);

    var accum_col = vec3<f32>(0.0);
    var trans = vec3<f32>(1.0);
    var in_fluid = false;
    let depth_norm_scale = max(dims.y - 2.0, 1.0);

    for (var b_iter = 0; b_iter < 112; b_iter = b_iter + 1) {
        if (bc.x < 0 || bc.y < 0 || bc.z < 0 || bc.x >= bdims.x || bc.y >= bdims.y || bc.z >= bdims.z) {
            break;
        }
        let t_brick_exit = min(min(t_max_v.x, min(t_max_v.y, t_max_v.z)), t_end);
        let occ = textureLoad(brick_tex, bc, 0);

        if (occ.g > 0.01 || occ.b > 0.01) {
            var cur_t = max(t, hit_box.x);
            while (cur_t < t_brick_exit) {
                let dt = clamp(0.22 + cur_t * 0.0025, 0.22, 0.45);
                let p = ro + rd * cur_t;
                let dm = sample_density(p);

                if (occ.g > 0.02 && dm.g > 0.12) {
                    let at = sample_albedo(p);
                    let is_waterfall = p.y > 4.2 && p.y < dims.y - 1.5;
                    var surf_shore = 0.0;
                    var surf_bed = 0.0;

                    if (!in_fluid && dm.g >= 0.26) {
                        in_fluid = true;
                        let fn_w = sample_fluid_normal(p, sim_t);
                        let ndv = clamp(dot(fn_w, -rd), 0.0, 1.0);
                        // Schlick Fresnel sky reflection
                        let fres = clamp(0.04 + 0.88 * pow(1.0 - ndv, 4.0), 0.04, 0.84);
                        let refl_dir = reflect(rd, fn_w);
                        let refl_sky = eval_sky(vec3<f32>(refl_dir.x, max(refl_dir.y, 0.05), refl_dir.z));
                        let sky_tint = select(vec3<f32>(0.80, 0.93, 1.12), vec3<f32>(1.0), style_id != 0);

                        // Dual-lobe crisp sun specular glints (sparkling diamond highlights + broad sun path)
                        let h_w = normalize(u.sun_dir.xyz - rd);
                        let ndh = max(dot(fn_w, h_w), 0.0);
                        let glint_sharp = pow(ndh, 110.0) * 3.4;
                        let glint_broad = pow(ndh, 24.0) * 0.56;
                        let sparkle_mask = 0.70 + 0.65 * vnoise2(p.xz * 2.8 + vec2<f32>(sim_t * 1.4, -sim_t * 2.2));
                        let sun_glint = (glint_sharp * sparkle_mask + glint_broad) * u.sun_col.rgb;

                        // Solid-adjacency shoreline & waterfall rim whitewater foam
                        let adj = sample_shore_adjacency(p);
                        surf_shore = adj.x;
                        surf_bed = adj.y;
                        let foam_n = vnoise2(p.xz * 2.3 + vec2<f32>(sim_t * 0.7, -sim_t * 1.8));
                        let shore_lap = smoothstep(0.14, 0.52, surf_shore) * smoothstep(0.28, 0.74, foam_n + surf_shore * 0.48);
                        let steep_wf = smoothstep(0.72, 0.35, abs(fn_w.y));
                        let wf_streak = 0.5 + 0.5 * sin(p.y * 3.8 + sim_t * 11.5 + p.x * 1.9 + foam_n * 3.2);
                        let surf_foam = select(
                            clamp(shore_lap * 0.92 + steep_wf * (0.45 + 0.50 * wf_streak), 0.0, 1.0),
                            0.0,
                            style_id != 0,
                        );

                        let surf_refl = refl_sky * sky_tint * 0.92 + sun_glint;
                        let foam_col = vec3<f32>(0.92, 0.97, 1.00) * (u.sun_col.rgb * 0.80 + u.sky_top.rgb * 0.62);
                        let refl_mix = mix(surf_refl * fres, foam_col * 0.86, surf_foam * 0.76);
                        accum_col += trans * refl_mix;
                        trans *= (1.0 - mix(fres * 0.62, 0.74, surf_foam));
                    }

                    // 4-Style fluid reinterpretation (0=Glacial Azure, 1=Molten Lava, 2=Arcane Plasma, 3=Emerald Cascade)
                    var fluid_rgb = at.rgb;
                    var fluid_glow = at.a;
                    var extra_foam = 0.0;
                    if (style_id == 1) {
                        let lava_pulse = 0.72 + 0.28 * sin(p.x * 0.7 + p.z * 0.9 - sim_t * 2.4);
                        fluid_rgb = mix(vec3<f32>(0.92, 0.22, 0.04), vec3<f32>(1.00, 0.76, 0.18), lava_pulse * select(0.55, 0.92, is_waterfall));
                        fluid_glow = max(fluid_glow, 0.72 + select(0.0, 0.22, is_waterfall));
                    } else if (style_id == 2) {
                        let arc_pulse = 0.70 + 0.30 * sin(p.z * 1.1 - p.y * 1.4 - sim_t * 3.0);
                        fluid_rgb = mix(vec3<f32>(0.08, 0.76, 1.00), vec3<f32>(0.68, 0.28, 1.00), arc_pulse * 0.45);
                        fluid_glow = max(fluid_glow, 0.56 + select(0.0, 0.28, is_waterfall));
                    } else if (style_id == 3) {
                        let jade_pulse = 0.68 + 0.32 * sin(p.x * 0.8 - p.y * 1.6 + sim_t * 2.6);
                        fluid_rgb = mix(vec3<f32>(0.06, 0.62, 0.46), vec3<f32>(0.36, 0.96, 0.74), jade_pulse * select(0.50, 0.88, is_waterfall));
                        fluid_glow = max(fluid_glow, 0.44 + select(0.0, 0.24, is_waterfall));
                    } else {
                        // Nordic glacial fjord water: deep sapphire channel -> turquoise shallows + animated ripple caustics
                        let c1 = vnoise2(p.xz * 1.15 - vec2<f32>(0.0, sim_t * 1.55));
                        let c2 = vnoise2(p.xz * 2.25 + vec2<f32>(sim_t * 0.85, -sim_t * 1.10) + vec2<f32>(6.4, -3.2));
                        let caustic = pow(clamp(1.0 - abs(c1 - c2) * 3.2, 0.0, 1.0), 3.0);
                        let deep_fjord = vec3<f32>(0.06, 0.24, 0.44);
                        let shallow_turq = vec3<f32>(0.14, 0.56, 0.72);
                        let shallow_w = clamp(surf_bed * 0.85 + surf_shore * 0.60, 0.0, 1.0);
                        let base_water = mix(deep_fjord, shallow_turq, shallow_w);
                        fluid_rgb = mix(fluid_rgb, base_water, 0.58) + vec3<f32>(0.16, 0.34, 0.42) * caustic * (0.45 + 0.55 * shallow_w);

                        if (is_waterfall) {
                            let streak = 0.5 + 0.5 * sin(p.y * 3.6 + sim_t * 11.5 + p.x * 2.1 + c1 * 2.4);
                            let crest = smoothstep(0.32, 0.86, streak * 0.75 + c2 * 0.40);
                            fluid_rgb = mix(vec3<f32>(0.34, 0.74, 0.94), vec3<f32>(0.96, 0.99, 1.00), 0.38 + crest * 0.60);
                            extra_foam = 0.45 + crest * 0.48;
                        }
                    }

                    let lin_w = srgb_to_linear(clamp(fluid_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
                    let foam = clamp(
                        max(smoothstep(0.50, 0.84, dot(fluid_rgb, vec3<f32>(0.3, 0.59, 0.11))), extra_foam),
                        0.0,
                        1.0,
                    );
                    let sigma = select(
                        mix(vec3<f32>(0.68, 0.20, 0.10), vec3<f32>(1.36, 1.30, 1.24), foam),
                        vec3<f32>(1.24, 1.14, 1.08),
                        style_id != 0
                    ) * dm.g;
                    let step_att = exp(-sigma * dt);
                    let sun_lit = u.sun_col.rgb * 0.74 + u.sky_top.rgb * 0.50;
                    var water_scatter = mix(lin_w * 0.90, vec3<f32>(0.92, 0.97, 1.00), foam * select(0.86, 0.16, style_id != 0)) * sun_lit;
                    if (fluid_glow > 0.05) {
                        water_scatter += blackbody_ramp(fluid_glow, lin_w, style_id) * 0.88;
                    }
                    accum_col += trans * (vec3<f32>(1.0) - step_att) * water_scatter;
                    trans *= step_att;
                } else {
                    in_fluid = false;
                }

                if (occ.b > 0.01 && dm.b > 0.015) {
                    let at = sample_albedo(p);
                    let is_hot_fire = at.a > 0.06;
                    var turb = 1.0;
                    if (is_hot_fire) {
                        let fn0 = vnoise2(vec2<f32>(p.x + p.z, p.y * 1.4 - sim_t * 3.8) * 1.25);
                        turb = 0.72 + 0.56 * fn0;
                    }
                    let ext = dm.b * 0.80 * turb * dt;
                    let step_t = exp(-ext);
                    let sun_vis = 0.55 + 0.45 * clamp(p.y / depth_norm_scale, 0.0, 1.0);
                    let lin_g = srgb_to_linear(at.rgb);
                    let scatter = lin_g * (u.sun_col.rgb * 0.62 * sun_vis + u.sky_horizon.rgb * 0.42);
                    var emit = vec3<f32>(0.0);
                    if (is_hot_fire) {
                        let core_temp = clamp(at.a * (0.85 + 0.35 * turb), 0.0, 1.0);
                        emit = blackbody_ramp(core_temp, lin_g, style_id) * 2.10;
                        let spark_p = floor(p * 1.85 + vec3<f32>(sin(sim_t * 1.7 + p.y) * 0.4, -sim_t * 5.2, cos(sim_t * 1.4 + p.x) * 0.4));
                        let spark = step(0.955, hash3(spark_p));
                        emit += vec3<f32>(2.6, 1.38, 0.30) * spark;
                    }
                    accum_col += trans * (1.0 - step_t) * (scatter + emit);
                    trans *= step_t;
                }

                if (max(trans.r, max(trans.g, trans.b)) < 0.02) {
                    break;
                }
                cur_t += dt;
            }
        }

        if (max(trans.r, max(trans.g, trans.b)) < 0.02 || t_brick_exit >= t_end) {
            break;
        }

        t = t_brick_exit + 0.0005;
        if (t_max_v.x < t_max_v.y) {
            if (t_max_v.x < t_max_v.z) {
                bc.x += step_i.x;
                t_max_v.x += t_delta.x;
            } else {
                bc.z += step_i.z;
                t_max_v.z += t_delta.z;
            }
        } else {
            if (t_max_v.y < t_max_v.z) {
                bc.y += step_i.y;
                t_max_v.y += t_delta.y;
            } else {
                bc.z += step_i.z;
                t_max_v.z += t_delta.z;
            }
        }
    }

    let hdr = accum_col + trans * bg_hdr;
    return vec4<f32>(aces_tonemap(hdr * u.sun_dir.w), 1.0);
}
