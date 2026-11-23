// Depth-Tested 3D Parametric Hero Knight & Live Projectile Orb Shader (WebGPU Linear HDR Output)
// Receives exact 3D-texture raymarched sun shadows, 3D-texture cave AO, and style-aware local emissive bounce.

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

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color_rough: vec4<f32>,
    @location(3) flags: vec4<f32>,
    @location(4) inst_pos_yaw: vec4<f32>,
    @location(5) inst_faction_scale: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) albedo_rough: vec4<f32>,
    @location(3) emissive: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let yaw = in.inst_pos_yaw.w;
    let c = cos(yaw);
    let s = sin(yaw);
    let sc = in.inst_faction_scale.w;

    let lp = in.pos * sc;
    let wp = vec3<f32>(
        lp.x * c + lp.z * s,
        lp.y,
        -lp.x * s + lp.z * c
    ) + in.inst_pos_yaw.xyz;

    let wn = normalize(vec3<f32>(
        in.normal.x * c + in.normal.z * s,
        in.normal.y,
        -in.normal.x * s + in.normal.z * c
    ));

    let base_col = mix(in.color_rough.rgb, in.color_rough.rgb * in.inst_faction_scale.rgb, in.flags.x);

    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.world_normal = wn;
    out.albedo_rough = vec4<f32>(base_col, in.color_rough.a);
    out.emissive = in.flags.y;
    return out;
}

fn sample_density(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(density_tex, vol_samp, p / u.vol_dims.xyz, 0.0);
}

fn sample_albedo(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(albedo_tex, vol_samp, p / u.vol_dims.xyz, 0.0);
}

fn blackbody_ramp(temp: f32, tint: vec3<f32>, style_id: i32) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
    if (style_id == 2) {
        let low = vec3<f32>(0.14, 0.78, 1.00);
        let mid = vec3<f32>(1.00, 0.18, 0.74);
        let hi  = vec3<f32>(1.00, 0.96, 0.80);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.35, 0.35) * (t * t * 3.8);
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

fn compute_sun_shadow_hero(p: vec3<f32>) -> f32 {
    let l = u.sun_dir.xyz;
    let dims = u.vol_dims.xyz;
    let bdims = vec3<i32>(u.brick_dims.xyz);
    let bsize = u.brick_dims.w;
    var t = 0.45;
    var vis = 1.0;
    for (var i = 0; i < 24; i = i + 1) {
        let sp = p + l * t;
        if (sp.y >= dims.y || sp.x <= 0.0 || sp.x >= dims.x || sp.z <= 0.0 || sp.z >= dims.z) {
            break;
        }
        let bc = clamp(vec3<i32>(floor(sp / bsize)), vec3<i32>(0), bdims - vec3<i32>(1));
        if (textureLoad(brick_tex, bc, 0).r < 0.04) {
            t += 2.2;
            continue;
        }
        let d = sample_density(sp).r;
        if (d >= 0.5) {
            return 0.0;
        }
        vis = min(vis, clamp((0.5 - d) * 4.5 + 0.15, 0.0, 1.0));
        t += 0.6 + t * 0.08;
    }
    return vis;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let n = normalize(in.world_normal);
    let v = normalize(u.cam_pos.xyz - p);
    let l = u.sun_dir.xyz;
    let h = normalize(l + v);
    let style_id = i32(round(u.sun_col.w));

    let lin_albedo = pow(clamp(in.albedo_rough.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
    if (in.emissive > 0.5) {
        let ndv_orb = max(dot(n, v), 0.0);
        let core = mix(vec3<f32>(1.0, 0.98, 0.92), lin_albedo, 0.55);
        let orb_hdr = core * (2.6 + 2.2 * pow(ndv_orb, 1.6)) + lin_albedo * 1.4;
        return vec4<f32>(orb_hdr, 1.0);
    }

    let ndl = max(dot(n, l), 0.0);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);
    let rough = in.albedo_rough.a;

    let shadow = compute_sun_shadow_hero(p + n * 0.2);
    let roof_occ = sample_density(p + vec3<f32>(0.0, 3.2, 0.0)).r * 0.45
        + sample_density(p + vec3<f32>(0.0, 5.2, 0.0)).r * 0.30;
    let ao = clamp(1.0 - roof_occ, 0.28, 1.0);

    var bounce = vec3<f32>(0.0);
    let offsets = array<vec3<f32>, 4>(
        vec3<f32>(-3.5, 0.2, 1.5),
        vec3<f32>(3.0, 0.2, 0.0),
        vec3<f32>(0.0, 0.2, 3.5),
        vec3<f32>(0.0, 0.2, -3.5)
    );
    for (var i = 0; i < 4; i = i + 1) {
        let s = sample_albedo(p + offsets[i]);
        if (s.a > 0.08) {
            bounce += blackbody_ramp(s.a, s.rgb, style_id) * 0.24;
        }
    }

    let sky_w = n.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.06, 0.08, 0.14), u.sky_top.rgb * 0.85 + u.sky_horizon.rgb * 0.30, sky_w) * ao;
    let shininess = mix(110.0, 12.0, rough);
    let spec_w = mix(0.95, 0.06, rough);
    let spec = pow(ndh, shininess) * spec_w * ndl * (0.25 + 0.75 * shadow);
    let rim = pow(1.0 - ndv, 3.2) * (u.sky_horizon.rgb * 0.28 * ao + bounce * 0.48);

    let hdr = lin_albedo * (u.sun_col.rgb * ndl * shadow + ambient + bounce)
        + u.sun_col.rgb * spec + rim;

    return vec4<f32>(hdr, 1.0);
}
