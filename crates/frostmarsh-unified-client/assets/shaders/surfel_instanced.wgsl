// Approach 4 GPU-Instanced Hard-Z Foliage & Flame Surfel Pass
// Prototypes (PineCanopy, SnowPine, MarshBrush, RiverReeds, WildfireFlame, SmokePlume) live in VRAM.
// Dynamic A+ live forest updates swap placement buffers in <0.5ms as wildfires burn trees away!

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

struct SplatProto {
    local_pos_rad: vec4<f32>,
    local_norm_flags: vec4<f32>,
    color_glow: vec4<f32>,
};

struct SplatPlacement {
    pos_yaw: vec4<f32>,
    tint_scale: vec4<f32>,
    lod_phase: vec4<f32>,
};

struct SplatRef {
    placement_idx: u32,
    proto_idx: u32,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<storage, read> protos: array<SplatProto>;
@group(0) @binding(2) var<storage, read> placements: array<SplatPlacement>;
@group(0) @binding(3) var<storage, read> draw_refs: array<SplatRef>;
@group(0) @binding(4) var shadow_tex: texture_depth_2d;
@group(0) @binding(5) var shadow_samp: sampler_comparison;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) delta_px: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

fn build_surfel_basis(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.92);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    var out: VsOut;
    let r = draw_refs[iid];
    let pl = placements[r.placement_idx];
    let pr = protos[r.proto_idx];

    let c = cos(pl.pos_yaw.w);
    let s = sin(pl.pos_yaw.w);
    let sc = pl.tint_scale.w;
    let lp = pr.local_pos_rad.xyz * sc;
    var rel_pos = pl.pos_yaw.xyz + vec3<f32>(lp.x * c - lp.z * s, lp.y, lp.x * s + lp.z * c);

    let flags = i32(round(pr.local_norm_flags.w));
    let two_sided = (flags & 1) != 0;
    let wind_sway = (flags & 2) != 0;

    if (wind_sway) {
        let t = u.origin_time.w;
        let h_factor = clamp(lp.y * 0.25, 0.0, 1.3);
        let sway = sin(t * 2.2 + pl.lod_phase.y + rel_pos.x * 0.35 + rel_pos.z * 0.25) * 0.14 * h_factor;
        rel_pos.x += sway;
        rel_pos.z += sway * 0.6;
    }

    let view_pos = u.view * vec4<f32>(rel_pos, 1.0);
    if (u.cam_fwd.w < 0.5 && view_pos.z > -0.3) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let ln = pr.local_norm_flags.xyz;
    var n = normalize(vec3<f32>(ln.x * c - ln.z * s, ln.y, ln.x * s + ln.z * c));
    let to_cam = select(u.cam_pos.xyz - rel_pos, -u.cam_fwd.xyz, u.cam_fwd.w > 0.5);
    let dist = max(length(u.cam_pos.xyz - rel_pos), 0.001);
    let view_dir = normalize(to_cam);
    let facing = dot(n, view_dir);

    if (!two_sided && facing < -0.20) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    if (two_sided && facing < 0.0) {
        n = -n;
    }

    let rad = pr.local_pos_rad.w * sc * pl.lod_phase.x;
    let B = build_surfel_basis(n);
    let M = mat3x3<f32>(B[0] * rad, B[1] * rad, B[2] * (rad * 0.18));
    let cov3d = M * transpose(M);

    let focal = u.cam_pos.w;
    var J: mat3x3<f32>;
    if (u.cam_fwd.w > 0.5) {
        J = mat3x3<f32>(vec3<f32>(focal, 0.0, 0.0), vec3<f32>(0.0, focal, 0.0), vec3<f32>(0.0, 0.0, 0.0));
    } else {
        let iz = -1.0 / view_pos.z;
        let lim = 1.3 * 0.5 * u.viewport.xy / focal;
        let tx = clamp(view_pos.x * iz, -lim.x, lim.x) / iz;
        let ty = clamp(view_pos.y * iz, -lim.y, lim.y) / iz;
        J = mat3x3<f32>(
            vec3<f32>(focal * iz, 0.0, 0.0),
            vec3<f32>(0.0, focal * iz, 0.0),
            vec3<f32>(focal * tx * iz * iz, focal * ty * iz * iz, 0.0),
        );
    }
    let V3 = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    let T = J * V3;
    let cov2d = T * cov3d * transpose(T);

    let a = cov2d[0][0] + 0.28;
    let b = cov2d[0][1];
    let c2 = cov2d[1][1] + 0.28;
    let det = a * c2 - b * b;
    if (det <= 1e-6) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let conic = vec3<f32>(c2, -b, a) / det;
    let ext = min(1.65 * sqrt(vec2<f32>(a, c2)), vec2<f32>(80.0));

    let clip = u.proj * view_pos;
    let ndc = clip.xy / clip.w;
    let ext_ndc = ext * 2.0 / u.viewport.xy;
    if (abs(ndc.x) - ext_ndc.x > 1.0 || abs(ndc.y) - ext_ndc.y > 1.0) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let alb = clamp(pr.color_glow.rgb * pl.tint_scale.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let glow = clamp(pr.color_glow.w * pl.lod_phase.z, 0.0, 1.0);

    var sh = 1.0;
    let lclip = u.light_vp * vec4<f32>(rel_pos + n * 0.25, 1.0);
    let lndc = lclip.xyz / lclip.w;
    let luv = vec2<f32>(lndc.x * 0.5 + 0.5, 0.5 - lndc.y * 0.5);
    if (luv.x > 0.01 && luv.x < 0.99 && luv.y > 0.01 && luv.y < 0.99 && lndc.z > 0.0 && lndc.z < 1.0) {
        let s_raw = textureSampleCompareLevel(shadow_tex, shadow_samp, luv, lndc.z - u.viewport.w * 1.5);
        sh = mix(1.0, select(s_raw, 0.55 + 0.45 * s_raw, two_sided), u.sun_col.w);
    }

    let fill = mix(u.bounce_col.rgb, u.sky_col.rgb, n.y * 0.5 + 0.5);
    let ndl = max(dot(n, u.sun_dir.xyz), 0.0) * sh;
    var light_scalar = ndl + 0.14 * max(facing, 0.0);
    if (u.viewport.z > 0.5) {
        light_scalar = round(light_scalar * u.viewport.z) / u.viewport.z;
    }
    var col = alb * (fill + u.sun_col.rgb * light_scalar) * u.sun_dir.w;
    let cool_ink = vec3<f32>(0.149, 0.118, 0.282);
    let warm_hi = vec3<f32>(1.000, 0.957, 0.839);
    let shadow_t = clamp(1.0 - light_scalar * 1.45, 0.0, 1.0);
    col = mix(col, cool_ink, shadow_t * 0.18);
    let hi_t = clamp((light_scalar - 0.78) * 1.8, 0.0, 1.0);
    col = mix(col, warm_hi, hi_t * 0.14);

    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lp_u = u.point_pos[k];
        if (lp_u.w > 0.0) {
            let d3 = lp_u.xyz - rel_pos;
            let rd = length(d3);
            if (rd < lp_u.w) {
                let att = max(0.0, 1.0 - rd / lp_u.w);
                col += (alb * 0.85 + vec3<f32>(0.10)) * u.point_col[k].rgb * att * att * 0.72;
            }
        }
    }

    if (glow > 0.05) { col = mix(col, alb * 1.52, glow); }
    let fog_t = 0.60 * smoothstep(u.sky_col.w, u.bounce_col.w, dist);
    col = mix(col, u.fog_col.rgb, fog_t * (1.0 - glow * 0.85));

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vid];
    out.delta_px = corner * ext;
    out.conic = conic;
    out.color = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    out.clip_pos = vec4<f32>(ndc + corner * ext_ndc, clip.z / clip.w, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let power = -0.5 * (in.conic.x * in.delta_px.x * in.delta_px.x + in.conic.z * in.delta_px.y * in.delta_px.y)
                - in.conic.y * in.delta_px.x * in.delta_px.y;
    if (power < -1.15) {
        discard;
    }
    return vec4<f32>(in.color.rgb, 1.0);
}
