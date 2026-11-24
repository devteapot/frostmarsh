// Wave 17B: Depth-Tested Living Volumetric Water, Waterfalls & Blackbody Fire/Ember Overlay (`Z=16..32`)
// Features:
//   - Animated multi-frequency directional flow wave normals & subsurface caustics
//   - Crisp Schlick Fresnel sky reflections + dual-lobe sparkling sun specular glints
//   - Solid-adjacency shoreline whitewater foam (hugging banks & bridge piers!) + rushing waterfall cascades
//   - Luminous turbulent fire/lava blackbody core + animated rising glowing ember sparks

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
@group(0) @binding(6) var scene_depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    var out: VsOut;
    let p = pos[vid];
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
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

// Multi-frequency animated Gerstner-like wave normal for sparkling living water & waterfalls
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
    // 4-octave animated flow ripples (strong enough to catch crisp sky reflections & sun diamond glints!)
    let flow_z = p.z - time * 1.9;
    let w1x = sin(p.x * 0.95 + flow_z * 1.55) * 0.095
            + sin(p.x * 2.35 - p.z * 1.25 + time * 2.6) * 0.055
            + cos(p.x * 4.70 + flow_z * 3.80 - time * 3.4) * 0.032;
    let w1z = cos(p.x * 1.30 - flow_z * 1.15) * 0.095
            + cos(p.x * 0.85 + p.z * 2.65 - time * 2.9) * 0.055
            + sin(p.x * 3.90 - flow_z * 4.40 + time * 3.1) * 0.032;
    return normalize(n + vec3<f32>(w1x, 0.0, w1z));
}

// Detect solid rock/bank/bridge-pier adjacency in `density_tex` for animated shoreline whitewater foam!
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
        return mix(bb, tint * 1.4, 0.35) * (t * t * 4.0);
    }
    if (style_id == 3) {
        let low = vec3<f32>(0.06, 0.68, 0.44);
        let mid = vec3<f32>(0.24, 1.00, 0.66);
        let hi  = vec3<f32>(0.86, 1.00, 0.92);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.35, 0.35) * (t * t * 3.8);
    }
    let low = vec3<f32>(0.98, 0.15, 0.02);
    let mid = vec3<f32>(1.00, 0.54, 0.08);
    let hi  = vec3<f32>(1.00, 0.96, 0.78);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    return mix(bb, tint * 1.35, 0.38) * (t * t * 4.2);
}

fn eval_sky(rd: vec3<f32>) -> vec3<f32> {
    let h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    let up = pow(clamp(rd.y, 0.0, 1.0), 0.52);
    var sky = mix(z16.sky_horizon.rgb * 1.18, z16.sky_top.rgb * 1.25, up);
    if (rd.y < 0.0) {
        sky = mix(z16.sky_horizon.rgb, z16.sky_horizon.rgb * 0.55, clamp(-rd.y * 4.0, 0.0, 1.0));
    }
    let sun_dot = max(dot(rd, u.sun_dir.xyz), 0.0);
    let sun_halo = pow(sun_dot, 12.0) * 0.42 + pow(sun_dot, 140.0) * 2.1;
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
    let p_near_h = z16.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let p_far_h = z16.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let p_near = p_near_h.xyz / p_near_h.w;
    let p_far = p_far_h.xyz / p_far_h.w;

    let ro_rel = select(u.cam_pos.xyz, p_near, u.cam_fwd.w > 0.5);
    let rd = normalize(p_far - p_near);
    let ro = ro_rel + u.origin_time.xyz + z16.world_offset.xyz;

    let style_id = i32(round(z16.vol_dims.w));
    let sim_t = z16.sky_top.w;

    var t_depth_max = 460.0;
    let has_geo = depth < 0.99999;
    if (has_geo) {
        let surf_h = z16.inv_vp * vec4<f32>(ndc, depth, 1.0);
        let surf_rel = surf_h.xyz / surf_h.w;
        t_depth_max = length(surf_rel - ro_rel);
    }

    let safe_rd = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let inv_rd = 1.0 / safe_rd;
    let dims = z16.vol_dims.xyz;
    let bdims = vec3<i32>(z16.brick_dims.xyz);
    let bsize = z16.brick_dims.w;
    let hit_box = intersect_aabb(ro, inv_rd, vec3<f32>(0.01), dims - vec3<f32>(0.01));

    let t_end = min(hit_box.y, t_depth_max);
    if (hit_box.x >= t_end) {
        discard;
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

    for (var b_iter = 0; b_iter < 144; b_iter = b_iter + 1) {
        if (bc.x < 0 || bc.y < 0 || bc.z < 0 || bc.x >= bdims.x || bc.y >= bdims.y || bc.z >= bdims.z) {
            break;
        }
        let t_brick_exit = min(min(t_max_v.x, min(t_max_v.y, t_max_v.z)), t_end);
        let occ = textureLoad(brick_tex, bc, 0);

        if (occ.g > 0.01 || occ.b > 0.01) {
            var cur_t = max(t, hit_box.x);
            while (cur_t < t_brick_exit) {
                let dt = clamp(0.26 + cur_t * 0.0030, 0.26, 0.52);
                let p = ro + rd * cur_t;
                let dm = sample_density(p);

                // --- Fluid volume: living glacial fjord river, abyssal lava & plunging 3D waterfalls ---
                let sky_cloud_condense = dims.y >= 48.0 && p.y > dims.y * 0.82 && dm.g < 0.36;
                if (occ.g > 0.02 && dm.g > 0.12 && !sky_cloud_condense) {
                    let at = sample_albedo(p); // Pre-styled by `neural_style_z16.wgsl`!
                    // Physically detect plunging waterfalls (open air below, not a river sitting on bedrock!)
                    let bed_deep = sample_density(p - vec3<f32>(0.0, 3.2, 0.0)).r;
                    let is_waterfall = p.y > dims.y * 0.22 && p.y < dims.y - 2.0
                        && (bed_deep < 0.22 || (dims.y <= 32.0 && p.y > dims.y * 0.24));
                    let is_abyss_magma = dims.y >= 48.0 && p.y < dims.y * 0.26 && at.a > 0.28;
                    let in_cloud_band = dims.y >= 48.0 && p.y > dims.y * 0.80;
                    var surf_shore = 0.0;
                    var surf_bed = 0.0;

                    if (!in_fluid && dm.g >= 0.26) {
                        in_fluid = true;
                        let fn_w = sample_fluid_normal(p, sim_t);
                        let ndv = clamp(dot(fn_w, -rd), 0.0, 1.0);
                        // Schlick Fresnel sky reflection
                        let fres = clamp(0.04 + 0.86 * pow(1.0 - ndv, 4.0), 0.04, 0.78);
                        let refl_dir = reflect(rd, fn_w);
                        let refl_sky = eval_sky(vec3<f32>(refl_dir.x, max(refl_dir.y, 0.05), refl_dir.z));
                        let sky_tint = select(vec3<f32>(0.78, 0.92, 1.12), vec3<f32>(1.0), style_id != 0);

                        // Dual-lobe crisp sun specular glints (sparkling diamond water highlights + sun path)
                        let h_w = normalize(u.sun_dir.xyz - rd);
                        let ndh = max(dot(fn_w, h_w), 0.0);
                        let glint_sharp = pow(ndh, 96.0) * 3.6;
                        let glint_broad = pow(ndh, 22.0) * 0.58;
                        let sparkle_mask = 0.70 + 0.65 * vnoise2(p.xz * 2.8 + vec2<f32>(sim_t * 1.4, -sim_t * 2.2));
                        let sun_glint = (glint_sharp * sparkle_mask + glint_broad) * u.sun_col.rgb;

                        // Solid-adjacency shoreline & waterfall rim whitewater foam!
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

                        let surf_refl = refl_sky * sky_tint * 0.88 + sun_glint;
                        let foam_col = vec3<f32>(0.92, 0.97, 1.00) * (u.sun_col.rgb * 0.82 + z16.sky_top.rgb * 0.65);
                        let refl_mix = mix(surf_refl * fres, foam_col * 0.85, surf_foam * 0.76);
                        accum_col += trans * refl_mix;
                        trans *= (1.0 - mix(fres * 0.58, 0.72, surf_foam));
                    }

                    // Rich Nordic fjord water coloration + animated caustics & waterfall whitewater!
                    var fluid_rgb = at.rgb;
                    var fluid_glow = at.a;
                    var extra_foam = 0.0;
                    if (style_id == 1 && !in_cloud_band) {
                        let lava_pulse = 0.72 + 0.28 * sin(p.x * 0.7 + p.z * 0.9 - sim_t * 2.4);
                        fluid_rgb = mix(fluid_rgb, vec3<f32>(0.96, 0.36, 0.06), lava_pulse * select(0.32, 0.56, is_waterfall));
                        fluid_glow = min(max(fluid_glow, 0.44 + select(0.0, 0.14, is_waterfall)), 0.62);
                    } else if (style_id == 2 && !in_cloud_band) {
                        let arc_pulse = 0.70 + 0.30 * sin(p.z * 1.1 - p.y * 1.4 - sim_t * 3.0);
                        fluid_rgb = mix(fluid_rgb, vec3<f32>(0.68, 0.28, 1.00), arc_pulse * 0.32);
                        fluid_glow = min(max(fluid_glow, 0.42 + select(0.0, 0.16, is_waterfall)), 0.62);
                    } else if (style_id == 3 && !in_cloud_band) {
                        let jade_pulse = 0.70 + 0.30 * sin(p.y * 2.2 - sim_t * 3.2 + p.x * 0.9);
                        fluid_rgb = mix(vec3<f32>(0.06, 0.42, 0.36), vec3<f32>(0.22, 0.88, 0.68), jade_pulse * select(0.32, 0.52, is_waterfall));
                        fluid_glow = min(max(fluid_glow * 0.55, 0.18 + select(0.0, 0.12, is_waterfall)), 0.38);
                    } else if (in_cloud_band) {
                        fluid_rgb = vec3<f32>(0.72, 0.78, 0.86);
                        fluid_glow = 0.0;
                    } else if (is_abyss_magma) {
                        // Glowing subterranean Abyssal Magma & Oil Rivers (`p.y < dims.y * 0.28`)!
                        let lava_pulse = 0.76 + 0.24 * sin(p.x * 0.65 + p.z * 0.85 - sim_t * 2.2);
                        fluid_rgb = mix(vec3<f32>(0.96, 0.34, 0.06), vec3<f32>(1.00, 0.82, 0.24), lava_pulse * 0.58);
                        fluid_glow = max(fluid_glow, 0.68);
                    } else {
                        // Nordic glacial river: deep sapphire channel -> turquoise shallows + animated ripple caustics
                        let c1 = vnoise2(p.xz * 1.15 - vec2<f32>(0.0, sim_t * 1.55));
                        let c2 = vnoise2(p.xz * 2.25 + vec2<f32>(sim_t * 0.85, -sim_t * 1.10) + vec2<f32>(6.4, -3.2));
                        let caustic = pow(clamp(1.0 - abs(c1 - c2) * 3.2, 0.0, 1.0), 3.0);
                        let deep_fjord = vec3<f32>(0.06, 0.24, 0.44);
                        let shallow_turq = vec3<f32>(0.14, 0.56, 0.72);
                        let shallow_w = clamp(surf_bed * 0.85 + surf_shore * 0.60, 0.0, 1.0);
                        let base_water = mix(deep_fjord, shallow_turq, shallow_w);
                        fluid_rgb = mix(fluid_rgb, base_water, 0.62) + vec3<f32>(0.16, 0.34, 0.42) * caustic * (0.45 + 0.55 * shallow_w);

                        // Waterfall plunging whitewater streaks + turbulent plunge-pool froth
                        if (is_waterfall) {
                            let streak = 0.5 + 0.5 * sin(p.y * 3.6 + sim_t * 11.5 + p.x * 2.1 + c1 * 2.4);
                            let crest = smoothstep(0.32, 0.86, streak * 0.75 + c2 * 0.40);
                            fluid_rgb = mix(vec3<f32>(0.34, 0.74, 0.94), vec3<f32>(0.96, 0.99, 1.00), 0.36 + crest * 0.62);
                            extra_foam = 0.45 + crest * 0.48;
                        }
                    }

                    let lin_w = srgb_to_linear(clamp(fluid_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
                    let foam = clamp(
                        max(smoothstep(0.54, 0.86, dot(fluid_rgb, vec3<f32>(0.3, 0.59, 0.11))), extra_foam),
                        0.0,
                        1.0,
                    );
                    let sigma = select(
                        mix(vec3<f32>(0.72, 0.22, 0.11), vec3<f32>(1.38, 1.32, 1.26), foam),
                        vec3<f32>(1.25, 1.15, 1.10),
                        style_id != 0
                    ) * dm.g;
                    let step_att = exp(-sigma * dt);
                    let sun_lit = u.sun_col.rgb * 0.78 + z16.sky_top.rgb * 0.52;
                    var water_scatter = mix(lin_w * 0.94, vec3<f32>(0.94, 0.98, 1.00), foam * select(0.86, 0.15, style_id != 0)) * sun_lit;
                    if (fluid_glow > 0.05) {
                        water_scatter += blackbody_ramp(fluid_glow, lin_w, style_id) * 0.90;
                    }
                    accum_col += trans * (vec3<f32>(1.0) - step_att) * water_scatter;
                    trans *= step_att;
                } else {
                    in_fluid = false;
                }

                // --- Gas volume: luminous wildfire flames, lava core, rising ember sparks & wispy clouds/steam ---
                if (occ.b > 0.01 && dm.b > 0.015) {
                    let at = sample_albedo(p);
                    let cloud_deck = select(0.0, smoothstep(dims.y * 0.74, dims.y * 0.88, p.y), dims.y >= 48.0);
                    let fire_thresh = select(0.08, 0.28, style_id == 1);
                    let is_hot_fire = at.a > fire_thresh && cloud_deck < 0.15;
                    var turb = 1.0;
                    if (is_hot_fire) {
                        let fn0 = vnoise2(vec2<f32>(p.x + p.z, p.y * 1.4 - sim_t * 3.8) * 1.25);
                        turb = 0.72 + 0.56 * fn0;
                    } else if (cloud_deck > 0.02) {
                        // Sculpt soft wispy drifting stratospheric clouds instead of blocky voxel curtains
                        let c_wisp = vnoise2(p.xz * 0.14 + vec2<f32>(sim_t * 0.08, -sim_t * 0.05)) * 0.65
                                   + vnoise2(p.xz * 0.34 - vec2<f32>(sim_t * 0.12, p.y * 0.25)) * 0.35;
                        turb = smoothstep(0.28, 0.78, c_wisp) * 0.32;
                    } else if (dims.y >= 48.0) {
                        // Keep tall Z=64 valley steam/mist airy and readable without blowing out the gorge
                        turb = 0.38;
                    }
                    let ext = dm.b * 0.72 * turb * dt;
                    let step_t = exp(-ext);
                    let sun_vis = 0.55 + 0.45 * clamp(p.y / max(dims.y - 2.0, 1.0), 0.0, 1.0);
                    let cloud_tint = select(
                        vec3<f32>(0.86, 0.92, 1.00),
                        vec3<f32>(0.42, 0.30, 0.28),
                        style_id == 1,
                    );
                    let lin_g = mix(srgb_to_linear(at.rgb), cloud_tint, cloud_deck * 0.78);
                    let cloud_boost = 1.0 + cloud_deck * select(0.25, 0.05, style_id == 1);
                    let scatter = lin_g * (u.sun_col.rgb * 0.60 * sun_vis + z16.sky_horizon.rgb * 0.40) * cloud_boost;
                    var emit = vec3<f32>(0.0);
                    if (is_hot_fire) {
                        let core_temp = clamp(at.a * (0.85 + 0.35 * turb), 0.0, 1.0);
                        emit = blackbody_ramp(core_temp, lin_g, style_id) * select(2.15, 1.45, style_id == 1);
                        // Animated rising ember sparks inside hot wildfire/lava plumes!
                        let spark_p = floor(p * 1.85 + vec3<f32>(sin(sim_t * 1.7 + p.y) * 0.4, -sim_t * 5.2, cos(sim_t * 1.4 + p.x) * 0.4));
                        let spark = step(0.955, hash3(spark_p));
                        emit += vec3<f32>(2.8, 1.45, 0.32) * spark;
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

    let alpha = clamp(1.0 - (trans.r + trans.g + trans.b) / 3.0, 0.0, 1.0);
    if (alpha < 0.012) {
        discard;
    }

    let unpremul_hdr = accum_col / max(alpha, 0.001);
    let vol_srgb = aces_tonemap(unpremul_hdr * z16.sky_horizon.w);

    if (!has_geo) {
        if (alpha < 0.22) {
            discard;
        }
        let sky_srgb = aces_tonemap(eval_sky(rd) * z16.sky_horizon.w);
        let blended = mix(sky_srgb, vol_srgb, alpha);
        return vec4<f32>(blended * 0.92, 0.92);
    }

    // Premultiplied alpha composite cleanly over the Hard-Z geometry in `ss_color_view`
    return vec4<f32>(vol_srgb * alpha, alpha);
}
