// Renderer A: Sparse 8x8x8 Brick DDA + Trilinear Smooth Isosurface Raymarching
// Zero hardcoded species rules — consumes ONLY density_mat (RGBA8 3D), albedo_thermal (RGBA8 3D),
// and the 8x8x8 coarse occupancy brick texture (RGBA8 3D).

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz = camera world pos, w = sim_time
    sun_dir: vec4<f32>,     // xyz = unit vector toward sun, w = exposure
    sun_col: vec4<f32>,     // rgb = directional sun radiance, w = focal_px
    sky_top: vec4<f32>,     // rgb = zenith sky, w = style_id
    sky_horizon: vec4<f32>, // rgb = horizon sky / alpine fog, w = stylized
    vol_dims: vec4<f32>,    // xyz = (256, 16, 256), w = tonemap_flag (1.0 = output tonemapped)
    brick_dims: vec4<f32>,  // xyz = (32, 2, 32), w = brick_size (8.0)
    viewport: vec4<f32>,    // xy = (w, h), zw = (1/w, 1/h)
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var density_tex: texture_3d<f32>;
@group(0) @binding(2) var albedo_tex: texture_3d<f32>;
@group(0) @binding(3) var brick_tex: texture_3d<f32>;
@group(0) @binding(4) var vol_samp: sampler;

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

fn micro_relief(p: vec3<f32>, rough: f32) -> vec3<f32> {
    // High-frequency data-driven surface micro-relief modulated purely by roughness_wetness
    let q = p * 3.1;
    let b = vec3<f32>(
        sin(q.x * 1.7 + cos(q.z * 2.3 + q.y * 1.1)),
        sin(q.y * 2.9 + cos(q.x * 1.5 - q.z * 1.9)),
        cos(q.z * 1.9 + sin(q.x * 2.1 + q.y * 1.7))
    );
    return b * (0.045 * rough);
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
    // Subtle physical capillary ripple on water surface
    let w1 = sin(p.x * 2.1 + p.z * 1.4 + time * 2.2) * 0.035;
    let w2 = cos(p.x * 1.6 - p.z * 2.5 + time * 1.8) * 0.035;
    return normalize(n + vec3<f32>(w1, 0.0, w2));
}

fn blackbody_ramp(temp: f32, tint: vec3<f32>) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
    let low = vec3<f32>(0.95, 0.14, 0.02);
    let mid = vec3<f32>(1.00, 0.52, 0.08);
    let hi  = vec3<f32>(1.00, 0.94, 0.72);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    // Blend blackbody physics with learned/authored thermal tint (allows cyan/violet cave crystals!)
    return mix(bb, tint * 1.35, 0.45) * (t * t * 3.8);
}

fn eval_sky(rd: vec3<f32>) -> vec3<f32> {
    let h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    let up = pow(clamp(rd.y, 0.0, 1.0), 0.55);
    var sky = mix(u.sky_horizon.rgb, u.sky_top.rgb, up);
    // Subtle lower horizon haze
    if (rd.y < 0.0) {
        sky = mix(u.sky_horizon.rgb, u.sky_horizon.rgb * 0.55, clamp(-rd.y * 4.0, 0.0, 1.0));
    }
    let sun_dot = max(dot(rd, u.sun_dir.xyz), 0.0);
    let sun_halo = pow(sun_dot, 14.0) * 0.32 + pow(sun_dot, 180.0) * 1.8;
    return sky + u.sun_col.rgb * sun_halo * smoothstep(-0.05, 0.18, h);
}

fn compute_ao_3d(p: vec3<f32>, n: vec3<f32>) -> f32 {
    // Build tangent basis around normal for 6-tap 3D-texture hemisphere AO -> deep cave interiors!
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
    // Extra overhead cave-roof occlusion check!
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
    // 100% data-driven 3D-texture local emissive gathering (lights cave walls from magma/crystals/fire!)
    var bounce = vec3<f32>(0.0);
    let dirs = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 0.2, 0.0),
        vec3<f32>(-1.0, 0.2, 0.0),
        vec3<f32>(0.0, 0.2, 1.0),
        vec3<f32>(0.0, 0.2, -1.0),
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0)
    );
    let p_off = p + n * 0.6;
    for (var i = 0; i < 6; i = i + 1) {
        let dir = normalize(dirs[i]);
        let s1 = sample_albedo(p_off + dir * 2.4);
        let s2 = sample_albedo(p_off + dir * 5.2);
        let w1 = max(dot(n, dir) * 0.5 + 0.5, 0.2);
        if (s1.a > 0.08) {
            bounce += blackbody_ramp(s1.a, s1.rgb) * w1 * 0.26;
        }
        if (s2.a > 0.08) {
            bounce += blackbody_ramp(s2.a, s2.rgb) * w1 * 0.14;
        }
    }
    // Also check slightly wider magma floor glow below ceilings/walls
    let s_low = sample_albedo(vec3<f32>(p.x, max(p.y - 3.6, 1.0), p.z));
    if (s_low.a > 0.10) {
        bounce += blackbody_ramp(s_low.a, s_low.rgb) * 0.22;
    }
    return bounce;
}

fn shade_solid_point(p: vec3<f32>, rd: vec3<f32>, hit_t: f32) -> vec3<f32> {
    let dm = sample_density(p);
    let at = sample_albedo(p);

    let rough = clamp(dm.a, 0.08, 1.0);
    let n_base = sample_solid_normal(p);
    let n = normalize(n_base + micro_relief(p, rough));
    let albedo = srgb_to_linear(at.rgb);
    let glow = at.a;

    let ao = compute_ao_3d(p, n_base);
    let shadow = compute_sun_shadow_3d(p, n_base);
    let l = u.sun_dir.xyz;
    let v = -rd;
    let ndl = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);

    // Specular Blinn-Phong scaled by roughness_wetness
    let shininess = mix(96.0, 10.0, rough);
    let spec_strength = mix(0.60, 0.03, rough);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let spec = pow(ndh, shininess) * spec_strength * (0.4 + 0.6 * fresnel) * ndl * shadow;

    // Sky hemisphere + cool bounce ambient, strongly modulated by 3D-texture AO!
    let sky_w = n.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.04, 0.06, 0.11), u.sky_top.rgb * 0.85 + u.sky_horizon.rgb * 0.30, sky_w) * ao;
    let direct = u.sun_col.rgb * ndl * shadow;
    let bounce_emit = compute_emissive_bounce_3d(p, n_base);

    var hdr = albedo * (direct + ambient * 0.72 + bounce_emit) + u.sun_col.rgb * spec;
    if (glow > 0.05) {
        hdr += blackbody_ramp(glow, srgb_to_linear(at.rgb));
    }

    // Subtle aerial perspective alpine mist
    let fog_t = clamp((hit_t - 52.0) / 195.0, 0.0, 0.78);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t);
    return hdr;
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

struct FragOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(in: VsOut) -> FragOut {
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let p_near_h = u.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let p_far_h = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let p_near = p_near_h.xyz / p_near_h.w;
    let p_far = p_far_h.xyz / p_far_h.w;

    let ro = u.cam_pos.xyz;
    let rd = normalize(p_far - p_near);
    let safe_rd = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let inv_rd = 1.0 / safe_rd;

    let dims = u.vol_dims.xyz;
    let bdims = vec3<i32>(u.brick_dims.xyz);
    let bsize = u.brick_dims.w;
    let hit_box = intersect_aabb(ro, inv_rd, vec3<f32>(0.01), dims - vec3<f32>(0.01));

    var out: FragOut;
    if (hit_box.x >= hit_box.y) {
        out.color = vec4<f32>(aces_tonemap(eval_sky(rd) * u.sun_dir.w), 1.0);
        out.depth = 1.0;
        return out;
    }

    var t = hit_box.x + 0.002;
    let t_end = min(hit_box.y, 280.0);

    // 3D Amanatides-Woo Brick DDA setup
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
    var hit_solid = false;
    var hit_t = t_end;
    var hit_p = ro;
    var in_fluid = false;

    for (var b_iter = 0; b_iter < 96; b_iter = b_iter + 1) {
        if (bc.x < 0 || bc.y < 0 || bc.z < 0 || bc.x >= bdims.x || bc.y >= bdims.y || bc.z >= bdims.z) {
            break;
        }
        let t_brick_exit = min(min(t_max_v.x, min(t_max_v.y, t_max_v.z)), t_end);
        let occ = textureLoad(brick_tex, bc, 0);

        if (occ.r > 0.01 || occ.g > 0.01 || occ.b > 0.01) {
            var cur_t = max(t, hit_box.x);
            var prev_t = cur_t;
            while (cur_t < t_brick_exit) {
                let dt = clamp(0.18 + cur_t * 0.0022, 0.18, 0.42);
                let p = ro + rd * cur_t;
                let dm = sample_density(p);

                // 1. Check smooth solid isosurface (solid_density == 0.5)
                if (occ.r > 0.02 && dm.r >= 0.5) {
                    var lo = prev_t;
                    var hi = cur_t;
                    for (var k = 0; k < 6; k = k + 1) {
                        let mid = 0.5 * (lo + hi);
                        if (sample_density(ro + rd * mid).r >= 0.5) {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    hit_t = 0.5 * (lo + hi);
                    hit_p = ro + rd * hit_t;
                    hit_solid = true;
                    break;
                }

                // 2. Translucent refractive fluid (river, glacial basin, plunging 3D waterfall)
                if (occ.g > 0.02 && dm.g > 0.12) {
                    let at = sample_albedo(p);
                    if (!in_fluid && dm.g >= 0.32) {
                        in_fluid = true;
                        let fn_w = sample_fluid_normal(p, u.cam_pos.w);
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
                    // Beer-Lambert colored water absorption + whitewater waterfall foam scatter!
                    let foam = smoothstep(0.45, 0.80, at.r);
                    let sigma = mix(vec3<f32>(0.55, 0.16, 0.09), vec3<f32>(1.25, 1.20, 1.15), foam) * dm.g;
                    let step_att = exp(-sigma * dt);
                    let sun_lit = u.sun_col.rgb * 0.68 + u.sky_top.rgb * 0.45;
                    let lin_w = srgb_to_linear(at.rgb);
                    let water_scatter = mix(lin_w * 0.72, vec3<f32>(0.88, 0.94, 1.00), foam) * sun_lit;
                    accum_col += trans * (vec3<f32>(1.0) - step_att) * water_scatter;
                    trans *= step_att;
                } else {
                    in_fluid = false;
                }

                // 3. Volumetric gas (wildfire flames, smoke plumes, waterfall plunge mist)
                if (occ.b > 0.01 && dm.b > 0.015) {
                    let at = sample_albedo(p);
                    let ext = dm.b * 0.78 * dt;
                    let step_t = exp(-ext);
                    let sun_vis = 0.55 + 0.45 * clamp(p.y / 14.0, 0.0, 1.0);
                    let lin_g = srgb_to_linear(at.rgb);
                    let scatter = lin_g * (u.sun_col.rgb * 0.62 * sun_vis + u.sky_horizon.rgb * 0.42);
                    var emit = vec3<f32>(0.0);
                    if (at.a > 0.05) {
                        emit = blackbody_ramp(at.a, lin_g) * 1.75;
                    }
                    accum_col += trans * (1.0 - step_t) * (scatter + emit);
                    trans *= step_t;
                }

                if (max(trans.r, max(trans.g, trans.b)) < 0.02) {
                    break;
                }
                prev_t = cur_t;
                cur_t += dt;
            }
        }

        if (hit_solid || max(trans.r, max(trans.g, trans.b)) < 0.02 || t_brick_exit >= t_end) {
            break;
        }

        // Advance brick DDA to next brick
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

    var hdr: vec3<f32>;
    if (hit_solid) {
        let solid_hdr = shade_solid_point(hit_p, rd, hit_t);
        hdr = accum_col + trans * solid_hdr;
        let clip = u.view_proj * vec4<f32>(hit_p, 1.0);
        out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    } else {
        hdr = accum_col + trans * eval_sky(rd);
        out.depth = 1.0;
    }

    out.color = vec4<f32>(aces_tonemap(hdr * u.sun_dir.w), 1.0);
    return out;
}
