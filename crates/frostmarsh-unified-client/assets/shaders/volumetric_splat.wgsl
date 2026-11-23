// Tier 3 Hybrid Depth-Tested Soft Volumetric 3D Gaussian Splatting Pass
// Renders leaping wildfire flames, billowing charcoal smoke, scalding steam clouds,
// elemental spell trails, and glowing marsh critters cleanly depth-tested against
// the shared Hard-Z terrain + hero + SDF surfel depth buffer (depth_write = false).

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

struct VolSplat {
    // xyz = camera-relative center, w = radius_xz
    pos_rad: vec4<f32>,
    // xyz = linear/emissive color, w = opacity
    color_alpha: vec4<f32>,
    // x = height_stretch (vertical elongation for flames), y = kind (0=fire,1=smoke,2=steam,3=wisp),
    // z = phase, w = wind_strength
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<storage, read> splats: array<VolSplat>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) @interpolate(flat) color_alpha: vec4<f32>,
    @location(2) @interpolate(flat) kind_phase: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    var out: VsOut;
    let sp = splats[iid];
    let t = u.origin_time.w;
    let kind = sp.params.y;
    let phase = sp.params.z;

    // Gentle rising buoyancy & wind drift animation per volumetric class
    var pos = sp.pos_rad.xyz;
    let wind_x = u.fog_col.w;
    if (kind < 0.5) {
        // Fire: licking upward oscillation + wind lean
        let flicker = sin(t * 9.5 + phase) * 0.16;
        pos.y += abs(sin(t * 5.2 + phase)) * 0.25;
        pos.x += flicker + wind_x * 0.22;
    } else if (kind < 1.5) {
        // Smoke: slow rising drift
        let cycle = fract(t * 0.22 + phase * 0.17);
        pos.y += cycle * 2.2;
        pos.x += cycle * wind_x * 1.4 + sin(t * 1.7 + phase) * 0.25;
        pos.z += cos(t * 1.4 + phase) * 0.20;
    } else if (kind < 2.5) {
        // Steam: expanding buoyant cloud
        let cycle = fract(t * 0.30 + phase * 0.23);
        pos.y += cycle * 1.6;
        pos.x += sin(t * 2.1 + phase) * 0.30 + cycle * wind_x * 0.8;
    } else {
        // Arcane / Critter wisp: gentle bob
        pos.y += sin(t * 3.4 + phase) * 0.18;
    }

    let view_pos = u.view * vec4<f32>(pos, 1.0);
    if (u.cam_fwd.w < 0.5 && view_pos.z > -0.35) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let rx = sp.pos_rad.w;
    let ry = rx * max(sp.params.x, 0.5);

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vid];

    // View-aligned anisotropic Gaussian billboard
    let vp_corner = vec4<f32>(
        view_pos.x + corner.x * rx * 1.9,
        view_pos.y + corner.y * ry * 1.9,
        view_pos.z,
        1.0,
    );
    let clip = u.proj * vp_corner;

    out.clip_pos = clip;
    out.local_uv = corner;
    out.color_alpha = sp.color_alpha;
    out.kind_phase = vec2<f32>(kind, phase);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.local_uv, in.local_uv);
    if (r2 > 1.0) {
        discard;
    }
    let kind = in.kind_phase.x;
    // Smooth 2D Gaussian kernel falloff (zero hard polygon edges!)
    let g = exp(-3.4 * r2);
    var rgb = in.color_alpha.rgb;
    var a = in.color_alpha.a * g;

    if (kind < 0.5) {
        // Fire: hot golden-cream core tapering to vibrant orange-crimson outer aura
        let core = exp(-7.5 * r2);
        let tip_fade = clamp(1.0 - max(in.local_uv.y, 0.0) * 0.45, 0.35, 1.0);
        rgb = mix(rgb, vec3<f32>(1.0, 0.94, 0.68), core * 0.72) * tip_fade;
        // Shoulder tonemap so overlapping fire splats stay richly saturated (no white washout!)
        rgb = rgb / (vec3<f32>(1.0) + rgb * 0.42);
        a = clamp(a * 0.92, 0.0, 0.88);
    } else if (kind < 1.5) {
        // Smoke: soft indigo-charcoal wisp
        a = clamp(a * 0.58, 0.0, 0.62);
    } else if (kind < 2.5) {
        // Steam: luminous cool-white mist
        a = clamp(a * 0.64, 0.0, 0.70);
    } else {
        // Bioluminescent critter / arcane glow
        let core = exp(-6.0 * r2);
        rgb = mix(rgb, vec3<f32>(1.0, 0.98, 0.90), core * 0.5);
        a = clamp(a * 0.85, 0.0, 0.88);
    }

    if (a < 0.015) {
        discard;
    }
    // Premultiplied alpha output
    return vec4<f32>(rgb * a, a);
}
