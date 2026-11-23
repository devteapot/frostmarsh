// Pass 2: Wave 12C Depth-Tested Volumetric Fluid & Gas Overlay + Sky + ACES Tonemap (WebGPU, Z=16..32)
// Reads hardware depth from Surface-Nets + Hero Mesh pass, skips all non-fluid/non-gas bricks
// in O(bricks) via 8x8x8 brick DDA, applies live 4-style fluid/plasma/lava/emerald appearance,
// and composites flowing water, plunging waterfall, fire, smoke & sky.

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
    let w1 = sin(p.x * 2.1 + p.z * 1.4 + time * 2.2) * 0.035;
    let w2 = cos(p.x * 1.6 - p.z * 2.5 + time * 1.8) * 0.035;
    return normalize(n + vec3<f32>(w1, 0.0, w2));
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
                    if (!in_fluid && dm.g >= 0.32) {
                        in_fluid = true;
                        let fn_w = sample_fluid_normal(p, sim_t);
                        let ndv = clamp(dot(fn_w, -rd), 0.0, 1.0);
                        let fres = clamp(0.04 + 0.92 * pow(1.0 - ndv, 4.5), 0.04, 0.92);
                        let refl_dir = reflect(rd, fn_w);
                        let refl_sky = eval_sky(vec3<f32>(refl_dir.x, max(refl_dir.y, 0.02), refl_dir.z));
                        let h_w = normalize(u.sun_dir.xyz - rd);
                        let sun_glint = pow(max(dot(fn_w, h_w), 0.0), 160.0) * 2.4;
                        let surf_refl = refl_sky * 0.95 + u.sun_col.rgb * sun_glint;
                        accum_col += trans * fres * surf_refl;
                        trans *= (1.0 - fres * 0.85);
                    }

                    // 4-Style fluid reinterpretation (0=Glacial Azure, 1=Molten Lava, 2=Arcane Plasma, 3=Emerald Cascade)
                    var fluid_rgb = at.rgb;
                    var fluid_glow = at.a;
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
                    } else if (is_waterfall) {
                        let streak = 0.5 + 0.5 * sin(p.y * 3.6 + sim_t * 9.0 + p.x * 1.8);
                        fluid_rgb = mix(vec3<f32>(0.42, 0.84, 0.98), vec3<f32>(0.92, 0.98, 1.00), 0.55 + streak * 0.35);
                    }

                    let foam = smoothstep(0.45, 0.80, dot(fluid_rgb, vec3<f32>(0.3, 0.59, 0.11)));
                    let sigma = select(
                        mix(vec3<f32>(0.55, 0.16, 0.09), vec3<f32>(1.25, 1.20, 1.15), foam),
                        vec3<f32>(1.22, 1.12, 1.06),
                        style_id != 0
                    ) * dm.g;
                    let step_att = exp(-sigma * dt);
                    let sun_lit = u.sun_col.rgb * 0.68 + u.sky_top.rgb * 0.45;
                    let lin_w = srgb_to_linear(fluid_rgb);
                    var water_scatter = mix(lin_w * 0.78, vec3<f32>(0.88, 0.94, 1.00), foam * select(0.85, 0.18, style_id != 0)) * sun_lit;
                    if (fluid_glow > 0.05) {
                        water_scatter += blackbody_ramp(fluid_glow, lin_w, style_id) * 0.85;
                    }
                    accum_col += trans * (vec3<f32>(1.0) - step_att) * water_scatter;
                    trans *= step_att;
                } else {
                    in_fluid = false;
                }

                if (occ.b > 0.01 && dm.b > 0.015) {
                    let at = sample_albedo(p);
                    let ext = dm.b * 0.78 * dt;
                    let step_t = exp(-ext);
                    let sun_vis = 0.55 + 0.45 * clamp(p.y / depth_norm_scale, 0.0, 1.0);
                    let lin_g = srgb_to_linear(at.rgb);
                    let scatter = lin_g * (u.sun_col.rgb * 0.62 * sun_vis + u.sky_horizon.rgb * 0.42);
                    var emit = vec3<f32>(0.0);
                    if (at.a > 0.05) {
                        emit = blackbody_ramp(at.a, lin_g, style_id) * 1.75;
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
