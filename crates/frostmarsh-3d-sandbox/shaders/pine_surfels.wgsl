// Iteration 9A: Hard-Z GPU-Instanced Sculpted Pine Canopy & Snow-Pine Surfel Pass
// Adapted from Frostmarsh's signature Approach 4 scalloped-tier conifer surfels!
// Placed cleanly along true 3D solid_density slopes & ridges, with live 3D-texture sun shadows,
// 9-step Cool-Indigo / Warm-Cream hue-shifted shading, gentle wind sway, and live Y-key style tinting!

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,      // xyz = cam_pos, w = sim_time
    sun_dir: vec4<f32>,      // xyz = sun_dir, w = exposure
    sun_col: vec4<f32>,      // rgb = sun_col, w = focal_px
    sky_top: vec4<f32>,      // rgb = sky_top, w = style_id (0, 1, 2)
    sky_horizon: vec4<f32>,  // rgb = sky_horizon, w = stylized (1.0 = Pixel-2xSS, 0.0 = Clean-3D)
    vol_dims: vec4<f32>,     // xyz = dims, w = tonemap_direct
    brick_dims: vec4<f32>,   // xyz = brick_dims, w = brick_size
    viewport: vec4<f32>,     // xy = render_w/h, zw = inv_w/h
};

struct SplatProto {
    local_pos_rad: vec4<f32>,    // xyz = local offset, w = radius
    local_norm_flags: vec4<f32>, // xyz = unit normal, w = flags (bit0=two_sided, bit1=wind, bit2=snow_cap, bit3=bark)
    color_glow: vec4<f32>,       // rgb = base nordic sRGB albedo, w = glow
};

struct SplatPlacement {
    pos_yaw: vec4<f32>,          // xyz = world base pos, w = yaw radians
    tint_scale: vec4<f32>,       // rgb = per-tree tint, w = uniform scale
    lod_phase: vec4<f32>,        // x = rad_mul, y = wind_phase, z = snow_factor, w = unused
};

struct SplatRef {
    placement_idx: u32,
    proto_idx: u32,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var density_tex: texture_3d<f32>;
@group(0) @binding(2) var albedo_tex: texture_3d<f32>;
@group(0) @binding(3) var brick_tex: texture_3d<f32>;
@group(0) @binding(4) var vol_samp: sampler;

@group(1) @binding(0) var<storage, read> protos: array<SplatProto>;
@group(1) @binding(1) var<storage, read> placements: array<SplatPlacement>;
@group(1) @binding(2) var<storage, read> draw_refs: array<SplatRef>;

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

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn eval_sky(rd: vec3<f32>) -> vec3<f32> {
    let up = pow(clamp(rd.y, 0.0, 1.0), 0.55);
    return mix(u.sky_horizon.rgb, u.sky_top.rgb, up);
}

fn style_transform_foliage(base_srgb: vec3<f32>, is_snow: bool, is_bark: bool, style_id: i32, snow_f: f32) -> vec4<f32> {
    if (style_id == 1) {
        // Style 1: VOLCANIC EMBER — charred obsidian bark, dark ash-pine boughs, glowing ember needles!
        if (is_bark) {
            return vec4<f32>(vec3<f32>(0.12, 0.09, 0.10), 0.04);
        }
        if (is_snow || snow_f > 0.5) {
            // Warm volcanic sulfur-ash dusted boughs with glowing ember tips
            let luma = dot(base_srgb, vec3<f32>(0.3, 0.59, 0.11));
            let col = mix(vec3<f32>(0.24, 0.14, 0.12), vec3<f32>(0.92, 0.54, 0.24), clamp(luma * 1.25, 0.0, 1.0));
            return vec4<f32>(col, select(0.10, 0.34, luma > 0.62));
        }
        let luma = dot(base_srgb, vec3<f32>(0.3, 0.59, 0.11));
        let col = mix(vec3<f32>(0.13, 0.09, 0.11), vec3<f32>(0.76, 0.30, 0.12), clamp(luma * 1.6, 0.0, 1.0));
        return vec4<f32>(col, select(0.02, 0.28, luma > 0.48));
    } else if (style_id == 2) {
        // Style 2: ARCANE THERMAL — bioluminescent frost-indigo & glowing neon-cyan pine needles!
        if (is_bark) {
            return vec4<f32>(vec3<f32>(0.10, 0.14, 0.32), 0.05);
        }
        let luma = dot(base_srgb, vec3<f32>(0.3, 0.59, 0.11));
        if (is_snow || snow_f > 0.5) {
            let col = mix(vec3<f32>(0.14, 0.42, 0.72), vec3<f32>(0.68, 0.98, 1.00), clamp(luma * 1.25, 0.0, 1.0));
            return vec4<f32>(col, select(0.14, 0.36, luma > 0.62));
        }
        let col = mix(vec3<f32>(0.08, 0.18, 0.44), vec3<f32>(0.10, 0.88, 0.86), clamp(luma * 1.75, 0.0, 1.0));
        return vec4<f32>(col, select(0.08, 0.28, luma > 0.44));
    }
    // Style 0: Signature Frostmarsh Nordic emerald & crisp alpine snow-pine
    return vec4<f32>(base_srgb, 0.0);
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
    var wp = pl.pos_yaw.xyz + vec3<f32>(lp.x * c - lp.z * s, lp.y, lp.x * s + lp.z * c);

    let flags = i32(round(pr.local_norm_flags.w));
    let two_sided = (flags & 1) != 0;
    let wind_sway = (flags & 2) != 0;
    let is_snow = (flags & 4) != 0;
    let is_bark = (flags & 8) != 0;

    if (wind_sway) {
        let t = u.cam_pos.w;
        let h_factor = clamp(lp.y * 0.18, 0.0, 1.2);
        let sway = sin(t * 2.1 + pl.lod_phase.y + wp.x * 0.28 + wp.z * 0.22) * 0.09 * h_factor;
        wp.x += sway;
        wp.z += sway * 0.6;
    }

    let view_pos = u.view * vec4<f32>(wp, 1.0);
    if (view_pos.z > -0.35) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let ln = pr.local_norm_flags.xyz;
    var n = normalize(vec3<f32>(ln.x * c - ln.z * s, ln.y, ln.x * s + ln.z * c));
    let to_cam = u.cam_pos.xyz - wp;
    let dist = max(length(to_cam), 0.001);
    let view_dir = to_cam / dist;
    let facing = dot(n, view_dir);

    if (!two_sided && facing < -0.22) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    if (two_sided && facing < 0.0) {
        n = -n;
    }

    let rad = pr.local_pos_rad.w * sc * pl.lod_phase.x;
    let B = build_surfel_basis(n);
    let M = mat3x3<f32>(B[0] * rad, B[1] * rad, B[2] * (rad * 0.22));
    let cov3d = M * transpose(M);

    let focal = u.sun_col.w;
    let iz = -1.0 / view_pos.z;
    let lim = 1.3 * 0.5 * u.viewport.xy / max(focal, 1.0);
    let tx = clamp(view_pos.x * iz, -lim.x, lim.x) / iz;
    let ty = clamp(view_pos.y * iz, -lim.y, lim.y) / iz;
    let J = mat3x3<f32>(
        vec3<f32>(focal * iz, 0.0, 0.0),
        vec3<f32>(0.0, focal * iz, 0.0),
        vec3<f32>(focal * tx * iz * iz, focal * ty * iz * iz, 0.0),
    );
    let V3 = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    let T = J * V3;
    let cov2d = T * cov3d * transpose(T);

    let a = cov2d[0][0] + 0.30;
    let b = cov2d[0][1];
    let c2 = cov2d[1][1] + 0.30;
    let det = a * c2 - b * b;
    if (det <= 1e-6) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let conic = vec3<f32>(c2, -b, a) / det;
    let ext = min(1.65 * sqrt(vec2<f32>(a, c2)), vec2<f32>(64.0));

    let clip = u.proj * view_pos;
    let ndc = clip.xy / clip.w;
    let ext_ndc = ext * 2.0 / u.viewport.xy;
    if (abs(ndc.x) - ext_ndc.x > 1.0 || abs(ndc.y) - ext_ndc.y > 1.0) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    // Style-aware albedo + emission (reacts live to Y-key style swap!)
    let style_id = i32(round(u.sky_top.w));
    let base_tinted = clamp(pr.color_glow.rgb * pl.tint_scale.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let st = style_transform_foliage(base_tinted, is_snow, is_bark, style_id, pl.lod_phase.z);
    let alb = srgb_to_linear(st.rgb);
    let glow = max(pr.color_glow.w, st.a);

    // Lightweight 3D-texture macro shadow + self-tier canopy AO
    let l = u.sun_dir.xyz;
    let occ_sun = textureSampleLevel(density_tex, vol_samp, (wp + l * 3.2) / u.vol_dims.xyz, 0.0).r;
    let sh = clamp(1.0 - smoothstep(0.35, 0.65, occ_sun) * 0.65, 0.35, 1.0);
    let ndl = max(dot(n, l), 0.0) * sh;
    var light_scalar = ndl + 0.14 * max(facing, 0.0);

    // 9-step Cool-Indigo / Warm-Cream stylized cel ramp when stylized mode is active
    let stylized = u.sky_horizon.w;
    if (stylized > 0.5) {
        let q_l = round(light_scalar * 9.0) / 9.0;
        light_scalar = mix(light_scalar, q_l, 0.58);
    }

    let sky_w = n.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.04, 0.06, 0.12), u.sky_top.rgb * 0.78 + u.sky_horizon.rgb * 0.32, sky_w);
    var hdr = alb * (u.sun_col.rgb * light_scalar + ambient * 0.75);

    if (stylized > 0.5) {
        let cool_ink = srgb_to_linear(vec3<f32>(0.149, 0.118, 0.282));
        let warm_hi = srgb_to_linear(vec3<f32>(1.000, 0.957, 0.839));
        let shadow_t = clamp(1.0 - light_scalar * 1.4, 0.0, 1.0);
        hdr = mix(hdr, cool_ink * 0.45, shadow_t * 0.20);
        let hi_t = clamp((light_scalar - 0.72) * 1.8, 0.0, 1.0);
        hdr = mix(hdr, warm_hi * 1.15, hi_t * 0.14);
    }

    if (glow > 0.02) {
        hdr += alb * glow * 2.2;
    }

    // Distance atmospheric perspective fog matching field_mesh.wgsl
    let rd = -view_dir;
    let fog_t = clamp((dist - 55.0) / 195.0, 0.0, 0.74);
    let fog_col = eval_sky(vec3<f32>(rd.x, max(rd.y, 0.04), rd.z));
    hdr = mix(hdr, fog_col, fog_t * fog_t * (1.0 - glow * 0.6));

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vid];
    out.delta_px = corner * ext;
    out.conic = conic;
    out.color = vec4<f32>(hdr, 1.0);
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
