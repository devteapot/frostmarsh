// Wave 17B: Tier 3 Hybrid Depth-Tested Soft Volumetric 3D Gaussian Splatting & AAA Combat VFX Pass
// Kinds (`params.y`):
//   0 = Wildfire Flame (incandescent golden core -> crimson-orange aura)
//   1 = Billowing Smoke (soft charcoal-indigo plume)
//   2 = Scalding Steam / Spray Mist
//   3 = Bioluminescent Wisp / Spell Glow
//   4 = High-Contrast 3D Diagonal Actuator Beam / Crystal Spark (white-hot core + vivid outer halo!)
//   5 = Hexagonal Energy-Grid Time-Dilation Dome Shell Patch (pulsing hex lattice + scan wave!)
//   6 = Soft Runic Ground Glow / Rising Thermal Updraft Swirl (shrine aura & Skywing updraft columns!)

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
    // x = height_stretch, y = kind (0..6), z = phase, w = wind_strength
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

// Crisp procedural 2D hexagonal grid edge distance (returns 1.0 on hex cell border, 0.0 inside)
fn hex_grid_edge(uv: vec2<f32>) -> f32 {
    let r = vec2<f32>(1.0, 1.7320508);
    let h = r * 0.5;
    let a = uv - r * floor(uv / r) - h;
    let b = (uv - h) - r * floor((uv - h) / r) - h;
    let gv = select(b, a, dot(a, a) < dot(b, b));
    let p = abs(gv);
    let edge_d = max(dot(p, normalize(r)), p.x);
    return smoothstep(0.40, 0.49, edge_d);
}

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

    var pos = sp.pos_rad.xyz;
    let wind_x = u.fog_col.w;
    if (kind < 0.5) {
        // 0: Fire — licking upward oscillation + wind lean
        let flicker = sin(t * 9.5 + phase) * 0.16;
        pos.y += abs(sin(t * 5.2 + phase)) * 0.25;
        pos.x += flicker + wind_x * 0.22;
    } else if (kind < 1.5) {
        // 1: Smoke — slow rising drift
        let cycle = fract(t * 0.22 + phase * 0.17);
        pos.y += cycle * 2.2;
        pos.x += cycle * wind_x * 1.4 + sin(t * 1.7 + phase) * 0.25;
        pos.z += cos(t * 1.4 + phase) * 0.20;
    } else if (kind < 2.5) {
        // 2: Steam — expanding buoyant cloud
        let cycle = fract(t * 0.30 + phase * 0.23);
        pos.y += cycle * 1.6;
        pos.x += sin(t * 2.1 + phase) * 0.30 + cycle * wind_x * 0.8;
    } else if (kind < 3.5) {
        // 3: Arcane / Critter wisp — gentle bob
        pos.y += sin(t * 3.4 + phase) * 0.18;
    } else if (kind < 4.5) {
        // 4: High-contrast actuator beam node / crystal spark — taut high-frequency energy crackle
        pos.y += sin(t * 14.0 + phase) * 0.04;
    } else if (kind < 5.5) {
        // 5: Hexagonal Time-Dilation Dome shell patch — calm harmonic shimmer
        pos.y += sin(t * 2.4 + phase) * 0.08;
    } else {
        // 6: Soft runic ground glow / rising thermal updraft swirl
        pos.y += sin(t * 3.0 + phase) * 0.10;
    }

    let view_pos = u.view * vec4<f32>(pos, 1.0);
    if (u.cam_fwd.w < 0.5 && view_pos.z > -0.35) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let rx = sp.pos_rad.w;
    let ry = rx * max(sp.params.x, 0.35);

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vid];

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
    let phase = in.kind_phase.y;
    let t = u.origin_time.w;

    let g = exp(-3.4 * r2);
    var rgb = in.color_alpha.rgb;
    var a = in.color_alpha.a * g;

    if (kind < 0.5) {
        // 0: Fire — hot golden-cream core tapering to vibrant orange-crimson outer aura
        let core = exp(-7.5 * r2);
        let tip_fade = clamp(1.0 - max(in.local_uv.y, 0.0) * 0.45, 0.35, 1.0);
        rgb = mix(rgb, vec3<f32>(1.0, 0.96, 0.72), core * 0.78) * tip_fade;
        rgb = rgb / (vec3<f32>(1.0) + rgb * 0.38);
        a = clamp(a * 0.94, 0.0, 0.90);
    } else if (kind < 1.5) {
        // 1: Smoke — soft indigo-charcoal wisp
        a = clamp(a * 0.56, 0.0, 0.60);
    } else if (kind < 2.5) {
        // 2: Steam — delicate translucent cool-white mist
        a = clamp(a * 0.42, 0.0, 0.40);
    } else if (kind < 3.5) {
        // 3: Bioluminescent critter / arcane glow
        let core = exp(-6.2 * r2);
        rgb = mix(rgb, vec3<f32>(1.0, 0.98, 0.92), core * 0.54);
        a = clamp(a * 0.86, 0.0, 0.88);
    } else if (kind < 4.5) {
        // 4: High-Contrast Luminous Actuator Beam / Crystal Spark
        // Razor-bright incandescent white-hot inner core + vivid saturated outer plasma/cryo/oil halo!
        let core = exp(-13.5 * r2);
        let halo = exp(-2.8 * r2);
        let pulse = 0.90 + 0.10 * sin(t * 16.0 + phase);
        rgb = mix(rgb * 1.15, vec3<f32>(1.0, 0.99, 0.94), core * 0.88) * pulse;
        rgb = rgb / (vec3<f32>(1.0) + rgb * 0.25);
        a = clamp(in.color_alpha.a * (halo * 0.62 + core * 0.92), 0.0, 0.94);
    } else if (kind < 5.5) {
        // 5: Hexagonal Energy-Grid Time-Dilation Dome Shell Patch!
        let hex = hex_grid_edge(in.local_uv * 3.2 + vec2<f32>(phase * 0.31, -t * 0.22));
        let scan = 0.5 + 0.5 * sin(in.local_uv.y * 9.0 - t * 4.2 + phase);
        let rim = smoothstep(0.45, 0.95, sqrt(r2)) * (1.0 - smoothstep(0.86, 1.0, sqrt(r2)));
        let mask = smoothstep(1.0, 0.25, r2);
        let lattice = clamp(hex * 0.85 + rim * 0.55 + scan * 0.14, 0.0, 1.0);
        rgb = mix(rgb * 0.85, vec3<f32>(0.96, 0.99, 1.0), hex * 0.52 + rim * 0.35);
        a = clamp(in.color_alpha.a * mask * (0.18 + lattice * 0.95), 0.0, 0.78);
    } else {
        // 6: Soft Additive-Feel Runic Ground Light / Swirling Thermal Updraft Column!
        let r = sqrt(r2);
        let ang = atan2(in.local_uv.y, in.local_uv.x);
        let swirl = 0.68 + 0.32 * sin(ang * 3.0 - r * 7.5 + t * 4.5 + phase);
        let ring = exp(-14.0 * (r - 0.52) * (r - 0.52));
        let soft = exp(-3.0 * r2);
        rgb = mix(rgb, vec3<f32>(1.0, 0.97, 0.88), ring * 0.42);
        a = clamp(in.color_alpha.a * (soft * 0.52 + ring * 0.58 * swirl) * (1.0 - smoothstep(0.84, 1.0, r)), 0.0, 0.72);
    }

    if (a < 0.014) {
        discard;
    }
    return vec4<f32>(rgb * a, a);
}
