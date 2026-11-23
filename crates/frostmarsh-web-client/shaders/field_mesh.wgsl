// Pass 1: Smooth Surface-Nets 3D Isosurface Mesh Shader (WebGPU)
// Samples albedo_thermal, roughness_wetness, analytical trilinear ∇solid_density normals,
// 3D-texture cave AO, 3D-texture raymarched sun shadows, and 3D-texture local emissive bounce
// directly from the two 256x16x256 RGBA8Unorm 3D textures.

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_col: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    vol_dims: vec4<f32>,
    brick_dims: vec4<f32>,
    viewport: vec4<f32>,
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

fn micro_relief(p: vec3<f32>, rough: f32) -> vec3<f32> {
    let q = p * 3.1;
    let b = vec3<f32>(
        sin(q.x * 1.7 + cos(q.z * 2.3 + q.y * 1.1)),
        sin(q.y * 2.9 + cos(q.x * 1.5 - q.z * 1.9)),
        cos(q.z * 1.9 + sin(q.x * 2.1 + q.y * 1.7))
    );
    return b * (0.045 * rough);
}

fn blackbody_ramp(temp: f32, tint: vec3<f32>) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
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
    let s_low = sample_albedo(vec3<f32>(p.x, max(p.y - 3.6, 1.0), p.z));
    if (s_low.a > 0.10) {
        bounce += blackbody_ramp(s_low.a, s_low.rgb) * 0.22;
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

    let shininess = mix(96.0, 10.0, rough);
    let spec_strength = mix(0.60, 0.03, rough);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let spec = pow(ndh, shininess) * spec_strength * (0.4 + 0.6 * fresnel) * ndl * shadow;

    let sky_w = n.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.04, 0.06, 0.11), u.sky_top.rgb * 0.85 + u.sky_horizon.rgb * 0.30, sky_w) * ao;
    let direct = u.sun_col.rgb * ndl * shadow;
    let bounce_emit = compute_emissive_bounce_3d(p, n_base);

    var hdr = albedo * (direct + ambient * 0.72 + bounce_emit) + u.sun_col.rgb * spec;
    if (glow > 0.05) {
        hdr += blackbody_ramp(glow, srgb_to_linear(at.rgb));
    }

    let fog_t = clamp((hit_t - 52.0) / 195.0, 0.0, 0.78);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t);

    return vec4<f32>(hdr, 1.0);
}
